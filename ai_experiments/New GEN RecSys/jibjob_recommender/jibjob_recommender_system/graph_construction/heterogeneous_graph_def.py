"""
Heterogeneous graph definition module.
Defines node types, edge types, and meta-paths for HeteroGCN.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Set

logger = logging.getLogger(__name__)

class HeterogeneousGraphDef:
    """
    Class responsible for defining the structure of heterogeneous graphs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HeterogeneousGraphDef with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.graph_config = config['graph']
        
        # Whether to include location nodes
        self.include_location_nodes = self.graph_config.get('include_location_nodes', True)
        
        # Define node types and edge types
        self.node_types = self._define_node_types()
        self.edge_types = self._define_edge_types()
        self.meta_paths = self._define_meta_paths()
        
    def _define_node_types(self) -> List[str]:
        """
        Define the node types in the heterogeneous graph.
        
        Returns:
            List[str]: List of node types.
        """
        node_types = ['professional', 'job', 'category']
        
        if self.include_location_nodes:
            node_types.append('location')
            
        logger.info(f"Defined node types: {node_types}")
        return node_types
        
    def _define_edge_types(self) -> List[Tuple[str, str, str]]:
        """
        Define the edge types in the heterogeneous graph.
        
        Returns:
            List[Tuple[str, str, str]]: List of (source_node_type, edge_name, target_node_type) tuples.
        """
        edge_types = [
            # Professional -> Category
            ('professional', 'interested_in', 'category'),
            # Category -> Professional (reverse)
            ('category', 'interest_of', 'professional'),
            # Job -> Category
            ('job', 'requires_category', 'category'),
            # Category -> Job (reverse)
            ('category', 'required_by', 'job'),
            # Professional -> Job (application)
            ('professional', 'applied_to', 'job'),
            # Job -> Professional (reverse)
            ('job', 'applied_by', 'professional'),
            # Professional -> Professional (similarity)
            ('professional', 'similar_profile_to', 'professional'),
            # Job -> Job (similarity)
            ('job', 'similar_description_to', 'job')
        ]
        
        if self.include_location_nodes:
            location_edges = [
                # Job -> Location
                ('job', 'located_at', 'location'),
                # Location -> Job (reverse)
                ('location', 'location_of', 'job'),
                # Professional -> Location
                ('professional', 'resides_in', 'location'),
                # Location -> Professional (reverse)
                ('location', 'residence_of', 'professional')
            ]
            edge_types.extend(location_edges)
            
        logger.info(f"Defined {len(edge_types)} edge types")
        return edge_types
        
    def _define_meta_paths(self) -> Dict[str, List[List[Tuple[str, str, str]]]]:
        """
        Define meta-paths for HeteroGCN message passing.
        A meta-path is a sequence of edge types that forms a path.
        
        Returns:
            Dict[str, List[List[Tuple[str, str, str]]]]: Dictionary of meta-path definitions.
        """
        # Define meta-paths for each node type
        meta_paths = {
            'professional': [
                # Professional -> Category -> Job
                [
                    ('professional', 'interested_in', 'category'),
                    ('category', 'required_by', 'job')
                ],
                # Professional -> Job (direct application)
                [
                    ('professional', 'applied_to', 'job')
                ],
                # Professional -> Professional (similarity)
                [
                    ('professional', 'similar_profile_to', 'professional')
                ]
            ],
            'job': [
                # Job -> Category -> Professional
                [
                    ('job', 'requires_category', 'category'),
                    ('category', 'interest_of', 'professional')
                ],
                # Job -> Professional (direct application)
                [
                    ('job', 'applied_by', 'professional')
                ],
                # Job -> Job (similarity)
                [
                    ('job', 'similar_description_to', 'job')
                ]
            ],
            'category': [
                # Category -> Professional -> Category
                [
                    ('category', 'interest_of', 'professional'),
                    ('professional', 'interested_in', 'category')
                ],
                # Category -> Job -> Category
                [
                    ('category', 'required_by', 'job'),
                    ('job', 'requires_category', 'category')
                ]
            ]
        }
        
        if self.include_location_nodes:
            # Add location-related meta-paths
            professional_loc_paths = [
                # Professional -> Location -> Job
                [
                    ('professional', 'resides_in', 'location'),
                    ('location', 'location_of', 'job')
                ]
            ]
            meta_paths['professional'].extend(professional_loc_paths)
            
            job_loc_paths = [
                # Job -> Location -> Professional
                [
                    ('job', 'located_at', 'location'),
                    ('location', 'residence_of', 'professional')
                ]
            ]
            meta_paths['job'].extend(job_loc_paths)
            
            location_paths = [
                # Location -> Job -> Location
                [
                    ('location', 'location_of', 'job'),
                    ('job', 'located_at', 'location')
                ],
                # Location -> Professional -> Location
                [
                    ('location', 'residence_of', 'professional'),
                    ('professional', 'resides_in', 'location')
                ]
            ]
            meta_paths['location'] = location_paths
            
        logger.info(f"Defined meta-paths for nodes: {list(meta_paths.keys())}")
        return meta_paths
        
    def get_node_types(self) -> List[str]:
        """
        Get the node types.
        
        Returns:
            List[str]: List of node types.
        """
        return self.node_types
        
    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """
        Get the edge types.
        
        Returns:
            List[Tuple[str, str, str]]: List of (source_node_type, edge_name, target_node_type) tuples.
        """
        return self.edge_types
        
    def get_meta_paths(self) -> Dict[str, List[List[Tuple[str, str, str]]]]:
        """
        Get the meta-paths.
        
        Returns:
            Dict[str, List[List[Tuple[str, str, str]]]]: Dictionary of meta-path definitions.
        """
        return self.meta_paths
        
    def get_message_types(self) -> Set[Tuple[str, str, str]]:
        """
        Get the unique message types for HeteroGCN.
        
        Returns:
            Set[Tuple[str, str, str]]: Set of unique message types.
        """
        message_types = set()
        
        for meta_paths_list in self.meta_paths.values():
            for meta_path in meta_paths_list:
                for edge_type in meta_path:
                    message_types.add(edge_type)
                    
        logger.info(f"Found {len(message_types)} unique message types")
        return message_types
