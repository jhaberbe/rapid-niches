import os
import json

import numpy as np
from tqdm import tqdm, trange

from scipy.spatial import cKDTree

import torch
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader



class PairwiseGraphBuilder:
    def __init__(self, X, locations, n_neighbors=10):
        if isinstance(locations, torch.Tensor):
            locations = locations.cpu().numpy()
        
        # Setup Data
        self.X = torch.tensor(X)
        self.locations = torch.tensor(locations)
        self.n_neighbors = n_neighbors
        self.ckd_tree = cKDTree(locations)

        # Compute nearest neighbors
        self._get_nearest_neighbors()

    def _get_nearest_neighbors(self):
        distances, indices = self.ckd_tree.query(self.locations, k=self.n_neighbors)
        self.indices = indices
        self.distances = distances
    
    def _fully_connected_graph(self, positions, include_self=True, device=None):
        if isinstance(positions, torch.Tensor):
            pos = positions
        else:
            pos = torch.tensor(positions, dtype=torch.float)

        N = pos.size(0)

        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(pos, pos, p=2)  # (N, N)

        # Get all possible edges (i,j)
        src, dst = torch.meshgrid(
            torch.arange(N, device=pos.device),
            torch.arange(N, device=pos.device),
            indexing='ij'
        )

        # Optionally remove self-edges
        if not include_self:
            mask = src != dst
            src, dst = src[mask], dst[mask]
            edge_attr = dist_matrix[mask].unsqueeze(1)
        else:
            src, dst = src.reshape(-1), dst.reshape(-1)
            edge_attr = dist_matrix.reshape(-1, 1)

        edge_index = torch.stack([src, dst], dim=0)

        if device is not None:
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)

        return edge_index, edge_attr
    
    def _construct_subgraph(self, index):
        # clone to avoid in-place edits
        X = self.X[self.indices[index]].clone()

        # Trying to avoid having the graph just pass information about the center node
        # Under global pooling conditions, I doubt it would ever actually leak, but I
        # also want to see if it would work if I only pass the center node's learned
        # embedding. 
        target = X[0]
        masked_X = X.clone()
        masked_X[0] = 0
        local_positions = self.locations[self.indices[index]]  # fixed typo
        edge_index, edge_attr = self._fully_connected_graph(local_positions)

        # I had some problems, now no more
        assert edge_index.max() < X.size(0), "Edge indices exceed node count!"

        return Data(
            X=masked_X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target=target[None, :]
        )
    
    def create_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(
            [self._construct_subgraph(i) for i in trange(self.X.shape[0])],
            batch_size=batch_size,
            shuffle=shuffle
        )



class GraphTransformerHandler:
    def __init__(self, model, optimizer=None, device=None):
        self.model = model
        self.optimizer = optimizer if optimizer else Adam(model.parameters(), lr=1e-2)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.training_history = {
            'batch_losses': [],
            'epoch_losses': []
        }
    
    def train(self, dataloader, num_epochs=10):
        """Train the model using the corrected training loop"""
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for i, batch in enumerate(loop):
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass through graph transformer
                context = self.model.graph_forward(batch)
                
                # Forward pass through normalizing flow
                log_prob = self.model.flow_forward(
                    batch.target,
                    context=context
                )
                
                # Calculate loss
                batch_loss = -log_prob.mean()

                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Track losses
                epoch_loss += batch_loss.item()
                self.training_history['batch_losses'].append(batch_loss.item())
                loop.set_postfix({"Batch Loss": f"{batch_loss.item():.4f}"})

            epoch_avg_loss = epoch_loss / len(dataloader)
            self.training_history['epoch_losses'].append(epoch_avg_loss)
            print(f"Epoch {epoch+1} Loss: {epoch_avg_loss:.4f}")
    
    def _create_unshuffled_dataloader(self, dataloader):
        """Temporarily create an unshuffled version of the dataloader"""
        # Get the original dataset from the dataloader
        dataset = dataloader.dataset
        
        # Create a new unshuffled dataloader with the same parameters
        unshuffled_dataloader = DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,  # Force no shuffling
            num_workers=getattr(dataloader, 'num_workers', 0),
            drop_last=False  # Don't drop last batch to preserve all data
        )
        
        return unshuffled_dataloader
    
    def predict_latent(self, dataloader, force_unshuffled=True):
        """Get latent representations using graph transformer head only
        
        Args:
            dataloader: DataLoader instance (can be shuffled)
            force_unshuffled: If True, creates temporary unshuffled dataloader
        """
        self.model.eval()
        latent_representations = []
        
        # Use unshuffled dataloader if requested
        if force_unshuffled:
            dataloader = self._create_unshuffled_dataloader(dataloader)
            print("Using temporary unshuffled dataloader for consistent latent representations")
        
        with torch.no_grad():
            loop = tqdm(dataloader, desc="Predicting Latent Representations")
            for i, batch in enumerate(loop):
                batch = batch.to(self.device)
                context = self.model.graph_forward(batch)
                latent_representations.append(context.detach().cpu().numpy())
        
        return np.vstack(latent_representations)
    
    def predict_with_flow(self, dataloader, context=None, force_unshuffled=True):
        """Get predictions using normalizing flow head
        
        Args:
            dataloader: DataLoader instance
            context: Optional pre-computed context tensor
            force_unshuffled: If True, creates temporary unshuffled dataloader
        """
        self.model.eval()
        flow_predictions = []
        
        # Use unshuffled dataloader if requested
        if force_unshuffled:
            dataloader = self._create_unshuffled_dataloader(dataloader)
            print("Using temporary unshuffled dataloader for consistent predictions")
        
        with torch.no_grad():
            loop = tqdm(dataloader, desc="Flow Predictions")
            for i, batch in enumerate(loop):
                batch = batch.to(self.device)
                
                # Use provided context or compute new one
                if context is None:
                    current_context = self.model.graph_forward(batch)
                else:
                    # If context is provided, we need to select the appropriate slice for this batch
                    batch_size = batch.target.shape[0]
                    start_idx = i * dataloader.batch_size
                    end_idx = start_idx + batch_size
                    current_context = context[start_idx:end_idx].to(self.device)
                
                # Get flow prediction
                log_prob = self.model.flow_forward(batch.target, context=current_context)
                flow_predictions.append(log_prob.detach().cpu().numpy())
        
        return np.concatenate(flow_predictions)
    
    def predict_both(self, dataloader, force_unshuffled=True):
        """Get both latent representations and flow predictions in one pass
        
        Args:
            dataloader: DataLoader instance
            force_unshuffled: If True, creates temporary unshuffled dataloader
        """
        self.model.eval()
        latent_representations = []
        flow_predictions = []
        
        # Use unshuffled dataloader if requested
        if force_unshuffled:
            dataloader = self._create_unshuffled_dataloader(dataloader)
            print("Using temporary unshuffled dataloader for consistent predictions")
        
        with torch.no_grad():
            loop = tqdm(dataloader, desc="Predicting Latent + Flow")
            for i, batch in enumerate(loop):
                batch = batch.to(self.device)
                
                # Get latent representations
                context = self.model.graph_forward(batch)
                latent_representations.append(context.detach().cpu().numpy())
                
                # Get flow predictions
                log_prob = self.model.flow_forward(batch.target, context=context)
                flow_predictions.append(log_prob.detach().cpu().numpy())
        
        return {
            'latent_representations': np.vstack(latent_representations),
            'flow_predictions': np.concatenate(flow_predictions)
        }
    
    def predict_flow_with_custom_context(self, targets, context):
        """Get flow predictions with custom context and targets"""
        self.model.eval()
        
        with torch.no_grad():
            # Ensure tensors are on correct device
            targets = torch.as_tensor(targets).to(self.device)
            context = torch.as_tensor(context).to(self.device)
            
            log_prob = self.model.flow_forward(targets, context=context)
            return log_prob.detach().cpu().numpy()
    
    def inference(self, batch_data):
        """Perform inference on a batch of data"""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(batch_data, 'to'):
                batch_data = batch_data.to(self.device)
            
            # Get graph transformer outputs
            context = self.model.graph_forward(batch_data)
            
            # Get flow outputs
            log_prob = self.model.flow_forward(batch_data.target, context=context)
            
            return {
                'context': context.cpu().numpy(),
                'log_prob': log_prob.cpu().numpy(),
                'loss': -log_prob.mean().cpu().numpy()
            }
    
    def save_checkpoint(self, filepath, include_optimizer=True):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_config': getattr(self.model, 'config', {})
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath, load_optimizer=True):
        """Load model checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file {filepath} not found")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {filepath}")
    
    def save_model(self, directory):
        """Save complete model and handler state"""
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(directory, 'model.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # Save handler state
        handler_state = {
            'training_history': self.training_history,
            'optimizer_state': self.optimizer.state_dict()
        }
        handler_path = os.path.join(directory, 'handler_state.pth')
        torch.save(handler_state, handler_path)
        
        # Save config
        config = {
            'device': str(self.device),
            'model_config': getattr(self.model, 'config', {})
        }
        config_path = os.path.join(directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {directory}")
    
    @classmethod
    def load_model(cls, directory, model_class, optimizer_class=Adam, optimizer_kwargs=None):
        """Load complete model and handler"""
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-3}
        
        # Load config
        config_path = os.path.join(directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        device = torch.device(config['device'])
        
        # Initialize model
        model_config = config.get('model_config', {})
        if model_config:
            model = model_class(**model_config)
        else:
            model = model_class()
            
        model_path = os.path.join(directory, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Initialize handler
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        handler = cls(model, optimizer, device)
        
        # Load handler state
        handler_state_path = os.path.join(directory, 'handler_state.pth')
        if os.path.exists(handler_state_path):
            handler_state = torch.load(handler_state_path, map_location=device)
            handler.training_history = handler_state['training_history']
            handler.optimizer.load_state_dict(handler_state['optimizer_state'])
        
        print(f"Model loaded from {directory}")
        return handler
    
    def get_training_history(self):
        """Get training history"""
        return self.training_history
    
    def set_model_mode(self, train=True):
        """Set model to training or evaluation mode"""
        if train:
            self.model.train()
        else:
            self.model.eval()
    
    def get_device(self):
        """Get the device the model is on"""
        return self.device