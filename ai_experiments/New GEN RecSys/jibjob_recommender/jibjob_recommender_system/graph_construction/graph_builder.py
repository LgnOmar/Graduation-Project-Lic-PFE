"""
Graph builder module for constructing graphs for GCN/HeteroGCN models.
"""

import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from torch_geometric.data import Data, HeteroData
from ..utils.helpers import create_id_mapping

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Class responsible for building graph data structures for GCN/HeteroGCN models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphBuilder with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.graph_config = config['graph']
        
        # Whether to use heterogeneous GCN
        self.use_heterogeneous = self.graph_config.get('use_heterogeneous', True)
        
        # Whether to include location nodes
        self.include_location_nodes = self.graph_config.get('include_location_nodes', True)
        
        # Node ID mappings
        self.professional_id_to_idx = {}
        self.job_id_to_idx = {}
        self.category_id_to_idx = {}
        self.location_id_to_idx = {}
        
        # Reverse mappings (for inference)
        self.professional_idx_to_id = {}
        self.job_idx_to_id = {}
        self.category_idx_to_id = {}
        self.location_idx_to_id = {}
        
    def build_heterogeneous_graph(self, data_dict: Dict[str, pd.DataFrame], features_dict: Dict[str, Any]) -> HeteroData:
        """
        Build a heterogeneous graph for HeteroGCN.
        
        Args:
            data_dict: Dictionary of DataFrames.
            features_dict: Dictionary of processed features.
            
        Returns:
            HeteroData: Heterogeneous graph data object.
        """
        logger.info("Building heterogeneous graph")
        
        # Create HeteroData object
        graph = HeteroData()
        
        # Create ID mappings
        self._create_mappings(data_dict, features_dict)
        
        # Add professional nodes
        professional_features = self._get_professional_features(features_dict)
        graph['professional'].x = professional_features
        graph['professional'].num_nodes = professional_features.shape[0]
        
        # Add job nodes
        job_features = self._get_job_features(features_dict)
        graph['job'].x = job_features
        graph['job'].num_nodes = job_features.shape[0]
        
        # Add category nodes
        category_features = self._get_category_features(features_dict)
        graph['category'].x = category_features
        graph['category'].num_nodes = category_features.shape[0]
        
        # Add location nodes if configured
        if self.include_location_nodes:
            location_features = self._get_location_features(features_dict)
            graph['location'].x = location_features
            graph['location'].num_nodes = location_features.shape[0]
            
        # Add edges
        # 1. Professional -> Category edges (INTERESTED_IN)
        prof_cat_edges = self._create_professional_category_edges(data_dict['professional_categories'])
        if prof_cat_edges[0].size(0) > 0:  # Check if there are any edges
            graph['professional', 'interested_in', 'category'].edge_index = prof_cat_edges
            
        # 2. Category -> Professional edges (INTEREST_OF)
        cat_prof_edges = (prof_cat_edges[1], prof_cat_edges[0])  # Reverse the edges
        if cat_prof_edges[0].size(0) > 0:  # Check if there are any edges
            graph['category', 'interest_of', 'professional'].edge_index = cat_prof_edges
            
        # 3. Job -> Category edges (REQUIRES_CATEGORY)
        job_cat_edges = self._create_job_category_edges(data_dict['jobs'])
        if job_cat_edges[0].size(0) > 0:  # Check if there are any edges
            graph['job', 'requires_category', 'category'].edge_index = job_cat_edges
            
        # 4. Category -> Job edges (REQUIRED_BY)
        cat_job_edges = (job_cat_edges[1], job_cat_edges[0])  # Reverse the edges
        if cat_job_edges[0].size(0) > 0:  # Check if there are any edges
            graph['category', 'required_by', 'job'].edge_index = cat_job_edges
            
        if self.include_location_nodes:
            # 5. Job -> Location edges (LOCATED_AT)
            job_loc_edges = self._create_job_location_edges(data_dict['jobs'])
            if job_loc_edges[0].size(0) > 0:  # Check if there are any edges
                graph['job', 'located_at', 'location'].edge_index = job_loc_edges
                
            # 6. Location -> Job edges (LOCATION_OF)
            loc_job_edges = (job_loc_edges[1], job_loc_edges[0])  # Reverse the edges
            if loc_job_edges[0].size(0) > 0:  # Check if there are any edges
                graph['location', 'location_of', 'job'].edge_index = loc_job_edges
                
            # 7. Professional -> Location edges (RESIDES_IN)
            prof_loc_edges = self._create_professional_location_edges(data_dict['users'])
            if prof_loc_edges[0].size(0) > 0:  # Check if there are any edges
                graph['professional', 'resides_in', 'location'].edge_index = prof_loc_edges
                
            # 8. Location -> Professional edges (RESIDENCE_OF)
            loc_prof_edges = (prof_loc_edges[1], prof_loc_edges[0])  # Reverse the edges
            if loc_prof_edges[0].size(0) > 0:  # Check if there are any edges
                graph['location', 'residence_of', 'professional'].edge_index = loc_prof_edges
                
        # 9. Professional -> Job edges (APPLIED_TO) if job applications data is available
        if 'job_applications' in data_dict:
            prof_job_edges, edge_attr = self._create_professional_job_edges(data_dict['job_applications'])
            if prof_job_edges[0].size(0) > 0:  # Check if there are any edges
                graph['professional', 'applied_to', 'job'].edge_index = prof_job_edges
                graph['professional', 'applied_to', 'job'].edge_attr = edge_attr
                
        # 10. Job -> Professional edges (APPLIED_BY) if job applications data is available
        if 'job_applications' in data_dict:
            job_prof_edges = (prof_job_edges[1], prof_job_edges[0])  # Reverse the edges
            if job_prof_edges[0].size(0) > 0:  # Check if there are any edges
                graph['job', 'applied_by', 'professional'].edge_index = job_prof_edges
                
        # 11. Professional -> Professional edges (SIMILAR_PROFILE_TO) based on content similarity
        prof_prof_edges = self._create_professional_similarity_edges(features_dict.get('professional_similarity_edges', []))
        if prof_prof_edges[0].size(0) > 0:  # Check if there are any edges
            graph['professional', 'similar_profile_to', 'professional'].edge_index = prof_prof_edges
            
        # 12. Job -> Job edges (SIMILAR_DESCRIPTION_TO) based on content similarity
        job_job_edges = self._create_job_similarity_edges(features_dict.get('job_similarity_edges', []))
        if job_job_edges[0].size(0) > 0:  # Check if there are any edges
            graph['job', 'similar_description_to', 'job'].edge_index = job_job_edges
            
        # Store mappings in the graph metadata for later use
        graph.professional_id_to_idx = self.professional_id_to_idx
        graph.job_id_to_idx = self.job_id_to_idx
        graph.category_id_to_idx = self.category_id_to_idx
        graph.location_id_to_idx = self.location_id_to_idx
        
        graph.professional_idx_to_id = self.professional_idx_to_id
        graph.job_idx_to_id = self.job_idx_to_id
        graph.category_idx_to_id = self.category_idx_to_id
        graph.location_idx_to_id = self.location_idx_to_id
            
        logger.info(f"Heterogeneous graph built with the following node types: {list(graph.node_types)}")
        logger.info(f"Edge types: {list(graph.edge_types)}")
        
        return graph
        
    def build_homogeneous_graph(self, data_dict: Dict[str, pd.DataFrame], features_dict: Dict[str, Any]) -> Data:
        """
        Build a homogeneous graph for GCN.
        
        Args:
            data_dict: Dictionary of DataFrames.
            features_dict: Dictionary of processed features.
            
        Returns:
            Data: Homogeneous graph data object.
        """
        logger.info("Building homogeneous graph")
        
        # Create Data object
        graph = Data()
        
        # Create ID mappings
        self._create_mappings(data_dict, features_dict)
        
        # Get features for different node types
        professional_features = self._get_professional_features(features_dict)
        job_features = self._get_job_features(features_dict)
        category_features = self._get_category_features(features_dict)
        
        # Combine features (ensure same dimensionality)
        feature_dim = professional_features.shape[1]
        job_features_padded = self._pad_features(job_features, feature_dim)
        category_features_padded = self._pad_features(category_features, feature_dim)
        
        # Combine all features
        node_features = torch.cat([
            professional_features,
            job_features_padded,
            category_features_padded
        ], dim=0)
        
        # Number of nodes of each type
        num_professionals = professional_features.shape[0]
        num_jobs = job_features.shape[0]
        num_categories = category_features.shape[0]
        
        # Add node features to the graph
        graph.x = node_features
        graph.num_nodes = node_features.shape[0]
        
        # Add node type indicators
        node_types = torch.zeros(graph.num_nodes, dtype=torch.long)
        node_types[:num_professionals] = 0  # Professional
        node_types[num_professionals:num_professionals+num_jobs] = 1  # Job
        node_types[num_professionals+num_jobs:] = 2  # Category
        graph.node_type = node_types
        
        # Create edge indices by offsetting the original node indices
        # 1. Professional -> Category edges
        prof_cat_edges = self._create_professional_category_edges(data_dict['professional_categories'])
        if prof_cat_edges[0].size(0) > 0:
            prof_cat_edges = (
                prof_cat_edges[0],
                prof_cat_edges[1] + num_professionals + num_jobs
            )
            
        # 2. Job -> Category edges
        job_cat_edges = self._create_job_category_edges(data_dict['jobs'])
        if job_cat_edges[0].size(0) > 0:
            job_cat_edges = (
                job_cat_edges[0] + num_professionals,
                job_cat_edges[1] + num_professionals + num_jobs
            )
            
        # 3. Professional -> Job edges (if job applications data is available)
        if 'job_applications' in data_dict:
            prof_job_edges, _ = self._create_professional_job_edges(data_dict['job_applications'])
            if prof_job_edges[0].size(0) > 0:
                prof_job_edges = (
                    prof_job_edges[0],
                    prof_job_edges[1] + num_professionals
                )
        else:
            prof_job_edges = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            
        # 4. Professional -> Professional similarity edges
        prof_prof_edges = self._create_professional_similarity_edges(features_dict.get('professional_similarity_edges', []))
            
        # 5. Job -> Job similarity edges
        job_job_edges = self._create_job_similarity_edges(features_dict.get('job_similarity_edges', []))
        if job_job_edges[0].size(0) > 0:
            job_job_edges = (
                job_job_edges[0] + num_professionals,
                job_job_edges[1] + num_professionals
            )
            
        # Combine all edges
        edge_indices = []
        for edges in [prof_cat_edges, job_cat_edges, prof_job_edges, prof_prof_edges, job_job_edges]:
            if edges[0].size(0) > 0:
                edge_indices.append(torch.stack(edges, dim=0))
                
        # Add reverse edges for undirected graph
        for edges in edge_indices.copy():
            edge_indices.append(torch.stack([edges[1], edges[0]], dim=0))
            
        # Combine all edges
        if edge_indices:
            graph.edge_index = torch.cat(edge_indices, dim=1)
        else:
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
            
        # Store mappings and offsets in the graph metadata for later use
        graph.professional_id_to_idx = self.professional_id_to_idx
        graph.job_id_to_idx = self.job_id_to_idx
        graph.category_id_to_idx = self.category_id_to_idx
        
        graph.professional_idx_to_id = self.professional_idx_to_id
        graph.job_idx_to_id = self.job_idx_to_id
        graph.category_idx_to_id = self.category_idx_to_id
        
        graph.num_professionals = num_professionals
        graph.num_jobs = num_jobs
        graph.num_categories = num_categories
        
        logger.info(f"Homogeneous graph built with {graph.num_nodes} nodes and {graph.edge_index.shape[1]} edges")
        
        return graph
        
    def _create_mappings(self, data_dict: Dict[str, pd.DataFrame], features_dict: Dict[str, Any]) -> None:
        """
        Create mappings from IDs to indices.
        
        Args:
            data_dict: Dictionary of DataFrames.
            features_dict: Dictionary of processed features.
        """
        # Professional ID to index mapping
        professional_ids = data_dict['users'][data_dict['users']['user_type'] == 'professional']['user_id'].tolist()
        self.professional_id_to_idx = create_id_mapping(professional_ids)
        self.professional_idx_to_id = {idx: id_str for id_str, idx in self.professional_id_to_idx.items()}
        
        # Job ID to index mapping
        job_ids = data_dict['jobs']['job_id'].tolist()
        self.job_id_to_idx = create_id_mapping(job_ids)
        self.job_idx_to_id = {idx: id_str for id_str, idx in self.job_id_to_idx.items()}
        
        # Category ID to index mapping
        category_ids = data_dict['categories']['category_id'].tolist()
        self.category_id_to_idx = create_id_mapping(category_ids)
        self.category_idx_to_id = {idx: id_str for id_str, idx in self.category_id_to_idx.items()}
        
        # Location ID to index mapping if locations are included
        if self.include_location_nodes:
            location_ids = data_dict['locations']['location_id'].tolist()
            self.location_id_to_idx = create_id_mapping(location_ids)
            self.location_idx_to_id = {idx: id_str for id_str, idx in self.location_id_to_idx.items()}
            
        logger.info(f"Created mappings for {len(self.professional_id_to_idx)} professionals, "
                   f"{len(self.job_id_to_idx)} jobs, {len(self.category_id_to_idx)} categories")
                   
        if self.include_location_nodes:
            logger.info(f"Created mappings for {len(self.location_id_to_idx)} locations")
            
    def _get_professional_features(self, features_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Get professional node features.
        
        Args:
            features_dict: Dictionary of processed features.
            
        Returns:
            torch.Tensor: Professional node features.
        """
        professional_features_dict = features_dict.get('professional_features', {})
        
        # Sort features by index
        sorted_features = [
            professional_features_dict[self.professional_idx_to_id[idx]]
            for idx in range(len(self.professional_idx_to_id))
            if idx in self.professional_idx_to_id and self.professional_idx_to_id[idx] in professional_features_dict
        ]
        
        # Check if we have any features
        if not sorted_features:
            logger.warning("No professional features found")
            # Return empty tensor with appropriate dimensions
            return torch.zeros((len(self.professional_id_to_idx), 1))
            
        # Stack tensors
        return torch.stack(sorted_features)
        
    def _get_job_features(self, features_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Get job node features.
        
        Args:
            features_dict: Dictionary of processed features.
            
        Returns:
            torch.Tensor: Job node features.
        """
        job_features_dict = features_dict.get('job_features', {})
        
        # Sort features by index
        sorted_features = [
            job_features_dict[self.job_idx_to_id[idx]]
            for idx in range(len(self.job_idx_to_id))
            if idx in self.job_idx_to_id and self.job_idx_to_id[idx] in job_features_dict
        ]
        
        # Check if we have any features
        if not sorted_features:
            logger.warning("No job features found")
            # Return empty tensor with appropriate dimensions
            return torch.zeros((len(self.job_id_to_idx), 1))
            
        # Stack tensors
        return torch.stack(sorted_features)
        
    def _get_category_features(self, features_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Get category node features.
        
        Args:
            features_dict: Dictionary of processed features.
            
        Returns:
            torch.Tensor: Category node features.
        """
        category_features_dict = features_dict.get('category_features', {})
        
        # Sort features by index
        sorted_features = [
            category_features_dict[self.category_idx_to_id[idx]]
            for idx in range(len(self.category_idx_to_id))
            if idx in self.category_idx_to_id and self.category_idx_to_id[idx] in category_features_dict
        ]
        
        # Check if we have any features
        if not sorted_features:
            logger.warning("No category features found")
            # Return empty tensor with appropriate dimensions
            return torch.zeros((len(self.category_id_to_idx), 1))
            
        # Stack tensors
        return torch.stack(sorted_features)
        
    def _get_location_features(self, features_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Get location node features.
        
        Args:
            features_dict: Dictionary of processed features.
            
        Returns:
            torch.Tensor: Location node features.
        """
        location_features_dict = features_dict.get('location_features', {})
        
        # Sort features by index
        sorted_features = [
            location_features_dict[self.location_idx_to_id[idx]]
            for idx in range(len(self.location_idx_to_id))
            if idx in self.location_idx_to_id and self.location_idx_to_id[idx] in location_features_dict
        ]
        
        # Check if we have any features
        if not sorted_features:
            logger.warning("No location features found")
            # Return empty tensor with appropriate dimensions
            return torch.zeros((len(self.location_id_to_idx), 2))
            
        # Stack tensors
        return torch.stack(sorted_features)
        
    def _pad_features(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Pad features to target dimension.
        
        Args:
            features: Feature tensor to pad.
            target_dim: Target feature dimension.
            
        Returns:
            torch.Tensor: Padded feature tensor.
        """
        current_dim = features.shape[1]
        
        if current_dim < target_dim:
            padding = torch.zeros((features.shape[0], target_dim - current_dim))
            return torch.cat([features, padding], dim=1)
        elif current_dim > target_dim:
            return features[:, :target_dim]
        else:
            return features
            
    def _create_professional_category_edges(self, professional_categories_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between professional nodes and category nodes.
        
        Args:
            professional_categories_df: DataFrame containing professional category mappings.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices (professional_indices, category_indices).
        """
        professional_indices = []
        category_indices = []
        
        for _, row in professional_categories_df.iterrows():
            professional_id = row['user_id']
            category_id = row['category_id']
            
            if professional_id in self.professional_id_to_idx and category_id in self.category_id_to_idx:
                professional_idx = self.professional_id_to_idx[professional_id]
                category_idx = self.category_id_to_idx[category_id]
                
                professional_indices.append(professional_idx)
                category_indices.append(category_idx)
                
        # Convert to tensors
        professional_indices = torch.tensor(professional_indices, dtype=torch.long)
        category_indices = torch.tensor(category_indices, dtype=torch.long)
        
        logger.info(f"Created {len(professional_indices)} professional-category edges")
        return professional_indices, category_indices
        
    def _create_job_category_edges(self, jobs_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between job nodes and category nodes.
        
        Args:
            jobs_df: DataFrame containing job data.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices (job_indices, category_indices).
        """
        job_indices = []
        category_indices = []
        
        # Check if required_category_id column exists
        if 'required_category_id' not in jobs_df.columns:
            logger.warning("required_category_id column not found in jobs data")
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
            
        for _, row in jobs_df.iterrows():
            job_id = row['job_id']
            category_id = row['required_category_id']
            
            if pd.isna(category_id) or category_id == '':
                continue
                
            if job_id in self.job_id_to_idx and category_id in self.category_id_to_idx:
                job_idx = self.job_id_to_idx[job_id]
                category_idx = self.category_id_to_idx[category_id]
                
                job_indices.append(job_idx)
                category_indices.append(category_idx)
                
        # Convert to tensors
        job_indices = torch.tensor(job_indices, dtype=torch.long)
        category_indices = torch.tensor(category_indices, dtype=torch.long)
        
        logger.info(f"Created {len(job_indices)} job-category edges")
        return job_indices, category_indices
        
    def _create_job_location_edges(self, jobs_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between job nodes and location nodes.
        
        Args:
            jobs_df: DataFrame containing job data.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices (job_indices, location_indices).
        """
        job_indices = []
        location_indices = []
        
        for _, row in jobs_df.iterrows():
            job_id = row['job_id']
            location_id = row['location_id']
            
            if pd.isna(location_id) or location_id == '':
                continue
                
            if job_id in self.job_id_to_idx and location_id in self.location_id_to_idx:
                job_idx = self.job_id_to_idx[job_id]
                location_idx = self.location_id_to_idx[location_id]
                
                job_indices.append(job_idx)
                location_indices.append(location_idx)
                
        # Convert to tensors
        job_indices = torch.tensor(job_indices, dtype=torch.long)
        location_indices = torch.tensor(location_indices, dtype=torch.long)
        
        logger.info(f"Created {len(job_indices)} job-location edges")
        return job_indices, location_indices
        
    def _create_professional_location_edges(self, users_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between professional nodes and location nodes.
        
        Args:
            users_df: DataFrame containing user data.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices (professional_indices, location_indices).
        """
        professional_indices = []
        location_indices = []
        
        # Filter for professional users
        professionals = users_df[users_df['user_type'] == 'professional']
        
        for _, row in professionals.iterrows():
            professional_id = row['user_id']
            location_id = row['location_id']
            
            if pd.isna(location_id) or location_id == '':
                continue
                
            if professional_id in self.professional_id_to_idx and location_id in self.location_id_to_idx:
                professional_idx = self.professional_id_to_idx[professional_id]
                location_idx = self.location_id_to_idx[location_id]
                
                professional_indices.append(professional_idx)
                location_indices.append(location_idx)
                
        # Convert to tensors
        professional_indices = torch.tensor(professional_indices, dtype=torch.long)
        location_indices = torch.tensor(location_indices, dtype=torch.long)
        
        logger.info(f"Created {len(professional_indices)} professional-location edges")
        return professional_indices, location_indices
        
    def _create_professional_job_edges(self, 
                                      job_applications_df: pd.DataFrame) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Create edges between professional nodes and job nodes.
        
        Args:
            job_applications_df: DataFrame containing job application data.
            
        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Edge indices (professional_indices, job_indices) and edge attributes.
        """
        professional_indices = []
        job_indices = []
        edge_attrs = []
        
        for _, row in job_applications_df.iterrows():
            professional_id = row['professional_user_id']
            job_id = row['job_id']
            
            if professional_id in self.professional_id_to_idx and job_id in self.job_id_to_idx:
                professional_idx = self.professional_id_to_idx[professional_id]
                job_idx = self.job_id_to_idx[job_id]
                
                # Edge attribute: positive interaction (1) or negative interaction (0)
                edge_attr = 1.0 if row.get('is_positive_interaction', True) else 0.0
                
                professional_indices.append(professional_idx)
                job_indices.append(job_idx)
                edge_attrs.append(edge_attr)
                
        # Convert to tensors
        professional_indices = torch.tensor(professional_indices, dtype=torch.long)
        job_indices = torch.tensor(job_indices, dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float).reshape(-1, 1)
        
        logger.info(f"Created {len(professional_indices)} professional-job edges from applications")
        return (professional_indices, job_indices), edge_attrs
        
    def _create_professional_similarity_edges(self, 
                                            similarity_edges: List[Tuple[str, str, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between similar professional nodes.
        
        Args:
            similarity_edges: List of (professional_id1, professional_id2, similarity) tuples.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices (professional_indices_src, professional_indices_dst).
        """
        professional_indices_src = []
        professional_indices_dst = []
        
        for prof_id1, prof_id2, similarity in similarity_edges:
            if prof_id1 in self.professional_id_to_idx and prof_id2 in self.professional_id_to_idx:
                prof_idx1 = self.professional_id_to_idx[prof_id1]
                prof_idx2 = self.professional_id_to_idx[prof_id2]
                
                professional_indices_src.append(prof_idx1)
                professional_indices_dst.append(prof_idx2)
                
        # Convert to tensors
        professional_indices_src = torch.tensor(professional_indices_src, dtype=torch.long)
        professional_indices_dst = torch.tensor(professional_indices_dst, dtype=torch.long)
        
        logger.info(f"Created {len(professional_indices_src)} professional-professional similarity edges")
        return professional_indices_src, professional_indices_dst
        
    def _create_job_similarity_edges(self, 
                                   similarity_edges: List[Tuple[str, str, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between similar job nodes.
        
        Args:
            similarity_edges: List of (job_id1, job_id2, similarity) tuples.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Edge indices (job_indices_src, job_indices_dst).
        """
        job_indices_src = []
        job_indices_dst = []
        
        for job_id1, job_id2, similarity in similarity_edges:
            if job_id1 in self.job_id_to_idx and job_id2 in self.job_id_to_idx:
                job_idx1 = self.job_id_to_idx[job_id1]
                job_idx2 = self.job_id_to_idx[job_id2]
                
                job_indices_src.append(job_idx1)
                job_indices_dst.append(job_idx2)
                
        # Convert to tensors
        job_indices_src = torch.tensor(job_indices_src, dtype=torch.long)
        job_indices_dst = torch.tensor(job_indices_dst, dtype=torch.long)
        
        logger.info(f"Created {len(job_indices_src)} job-job similarity edges")
        return job_indices_src, job_indices_dst
