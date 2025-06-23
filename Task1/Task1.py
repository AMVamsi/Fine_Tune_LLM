"""
MedCAT SNOMED CT Entity Extraction and Neo4j Visualization
================================================================

This module provides a streamlined pipeline for:
1. Loading MedCAT models
2. Extracting medical entities from text
3. Storing entities in Neo4j using RDF
4. Basic visualization and analysis

Author: Mohan Adluru 
Created: June 2025
"""
#install necessary packages
# %pip install rdflib neo4j rdflib-neo4j
# %pip install medcat~=1.14.0
# %pip install pandas

# Import necessary libraries

import pandas as pd
import os
from typing import Dict, List, Any


# =============================================================================
# SECTION 1: MEDCAT SETUP AND ENTITY EXTRACTION
# =============================================================================

class MedCATProcessor:
    """Handles MedCAT model loading and entity extraction."""
    
    def __init__(self, model_path: str):
        """Initialize with MedCAT model path."""
        self.model_path = model_path
        self.cat = None
        self._load_model()
    
    def _load_model(self):
        """Load the MedCAT model."""
        try:
            from medcat.cat import CAT
            self.cat = CAT.load_model_pack(self.model_path)
            # Set lower threshold for better entity detection
            self.cat.config.linking['min_acc'] = 0.1
            print(f"MedCAT model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading MedCAT model: {e}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract medical entities from text."""
        if not self.cat:
            raise ValueError("MedCAT model not loaded")
        
        entities = self.cat.get_entities(text)
        return entities
    
    def process_to_dataframe(self, entities: Dict[str, Any]) -> pd.DataFrame:
        """Convert extracted entities to structured DataFrame."""
        extracted_entities = []
        
        for key, entity in entities["entities"].items():
            # Only include SNOMED CT entities
            if "SCTID" in entity.get("cui", ""):
                extracted_entities.append({
                    "CUI": entity["cui"],
                    "Entity": entity["pretty_name"],
                    "Start": entity["start"],
                    "End": entity["end"],
                    "Confidence": entity["acc"]
                })
        
        return pd.DataFrame(extracted_entities)


# =============================================================================
# SECTION 2: NEO4J CONNECTION AND RDF MANAGEMENT
# =============================================================================

class Neo4jRDFManager:
    """Manages Neo4j connections and RDF graph operations."""
    
    def __init__(self, neo4j_uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection parameters."""
        self.neo4j_uri = neo4j_uri
        self.username = username
        self.password = password
        self.database = database
        self.graph = None
        self.snomed_namespace = None
        
    def setup_rdf_graph(self):
        """Setup RDF graph with Neo4j store."""
        try:
            from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
            from rdflib import Graph, Namespace
            
            # Setup authentication
            auth_data = {
                'uri': self.neo4j_uri,
                'database': self.database,
                'user': self.username,
                'pwd': self.password
            }
            
            # Configure Neo4j store
            config = Neo4jStoreConfig(
                auth_data=auth_data,
                handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
                batching=True
            )
            
            # Create RDF graph
            self.graph = Graph(store=Neo4jStore(config=config))
            self.snomed_namespace = Namespace("http://snomed.info/id/")
            
            print("RDF graph connected to Neo4j successfully")
            
        except Exception as e:
            print(f"Error setting up RDF graph: {e}")
            raise
    
    def add_entities_to_graph(self, entities_df: pd.DataFrame):
        """Add medical entities to RDF graph."""
        if self.graph is None:
            raise ValueError("RDF graph not initialized. Call setup_rdf_graph() first.")
        
        from rdflib import URIRef, Literal
        from rdflib.namespace import RDF, RDFS
        
        for _, entity in entities_df.iterrows():
            entity_uri = URIRef(self.snomed_namespace[entity["CUI"]])
            
            # Add entity as class with label
            self.graph.add((entity_uri, RDF.type, RDFS.Class))
            self.graph.add((entity_uri, RDFS.label, Literal(entity["Entity"])))
            
            # Add confidence as property
            confidence_prop = URIRef(self.snomed_namespace["confidence"])
            self.graph.add((entity_uri, confidence_prop, Literal(entity["Confidence"])))
        
        print(f"Added {len(entities_df)} entities to RDF graph")
    
    def add_sample_relationships(self, entities_df: pd.DataFrame):
        """Add sample relationships between entities."""
        if len(entities_df) < 2:
            print("Need at least 2 entities to create relationships")
            return
        
        from rdflib import URIRef
          # Create a simple relationship between first two entities
        entity1_uri = URIRef(self.snomed_namespace[entities_df.iloc[0]["CUI"]])
        entity2_uri = URIRef(self.snomed_namespace[entities_df.iloc[1]["CUI"]])
        relation_uri = URIRef(self.snomed_namespace["is_related_to"])
        
        self.graph.add((entity1_uri, relation_uri, entity2_uri))
        print("Added sample relationship between entities")
    
    def commit_to_neo4j(self):
        """Commit RDF graph to Neo4j database."""
        if self.graph:
            self.graph.close(True)
            print("RDF graph committed to Neo4j database")
        else:
            print("No graph to commit")


# =============================================================================
# SECTION 3: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    # Configuration for MedCAT and Neo4j
    MODEL_PATH = r"C:/Vamsi/Study/SS25/Track 2/Exercise/mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5.zip"
    
    NEO4J_CONFIG = {
        "neo4j_uri": "neo4j+s://78ce48d5.databases.neo4j.io",
        "username": "neo4j",
        "password": "REMOVED_FOR_SECURITY",
        "database": "neo4j"
    }
    
    try:
        print("Starting MedCAT-Neo4j Pipeline")
        print("="*60)
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found: {MODEL_PATH}")
            print("Please update MODEL_PATH with the correct path to your MedCAT model.")
            return
        
        # Initialize components (mirroring notebook structure)
        processor = MedCATProcessor(MODEL_PATH)
        neo4j_manager = Neo4jRDFManager(**NEO4J_CONFIG)
        neo4j_manager.setup_rdf_graph()
        
        # Test text from original notebook
        text = "Patient was diagnosed with chronic kidney disease and hypertension."
        print(f"Processing text: {text}")
        
        # Extract entities (like notebook cell 12)
        entities = processor.extract_entities(text)
        entities_df = processor.process_to_dataframe(entities)
        
        # Print results (like original notebook)
        print("Extracted Entities DataFrame:")
        print(entities_df)
        
        # Store in Neo4j (like notebook cells 16-17)
        if not entities_df.empty:
            print("Storing entities in Neo4j...")
            neo4j_manager.add_entities_to_graph(entities_df)
            neo4j_manager.add_sample_relationships(entities_df)
            neo4j_manager.commit_to_neo4j()
            print("MedCAT entities successfully stored in Neo4j!")
        else:
            print("No entities found to store.")
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        print("Please check your configuration and dependencies.")


if __name__ == "__main__":
    main()
