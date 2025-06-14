FROM neo4j:5.15-community

# Set environment variables for Neo4j configuration
ENV NEO4J_AUTH=neo4j/matters2025
ENV NEO4J_PLUGINS=["apoc"]
ENV NEO4J_apoc_export_file_enabled=true
ENV NEO4J_apoc_import_file_enabled=true
ENV NEO4J_apoc_import_file_use__neo4j__config=true

# Enable vector search capabilities
ENV NEO4J_dbms_security_procedures_unrestricted=apoc.*
ENV NEO4J_dbms_security_procedures_allowlist=apoc.*

# Configure memory and performance
ENV NEO4J_dbms_memory_heap_initial__size=512m
ENV NEO4J_dbms_memory_heap_max__size=1G
ENV NEO4J_dbms_memory_pagecache_size=512m

# Expose ports
EXPOSE 7474 7687

# Use the default entrypoint from the Neo4j image
CMD ["neo4j"]