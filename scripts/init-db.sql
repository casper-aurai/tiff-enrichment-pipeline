-- =============================================================================
-- TIFF Enrichment Pipeline - Database Initialization
-- This script sets up the PostGIS database with required extensions and schemas
-- =============================================================================

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS postgis_raster;

-- Create schemas for organizing tables
CREATE SCHEMA IF NOT EXISTS pipeline;
CREATE SCHEMA IF NOT EXISTS metadata;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path to include our schemas
ALTER DATABASE tiff_pipeline SET search_path TO pipeline, metadata, monitoring, public;

-- Create roles and permissions
DO $$ 
BEGIN
    -- Create read-only role for reporting
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pipeline_readonly') THEN
        CREATE ROLE pipeline_readonly;
    END IF;
    
    -- Create processing role with limited permissions
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pipeline_processor') THEN
        CREATE ROLE pipeline_processor;
    END IF;
END $$;

-- Grant schema usage
GRANT USAGE ON SCHEMA pipeline TO pipeline_readonly, pipeline_processor;
GRANT USAGE ON SCHEMA metadata TO pipeline_readonly, pipeline_processor;
GRANT USAGE ON SCHEMA monitoring TO pipeline_readonly, pipeline_processor;

-- Enable row-level security (RLS) - for future multi-tenant support
ALTER DATABASE tiff_pipeline SET row_security = on;

-- Create custom data types
CREATE TYPE pipeline.processing_status AS ENUM (
    'pending',
    'processing', 
    'completed',
    'failed',
    'retrying',
    'cancelled'
);

CREATE TYPE pipeline.file_type AS ENUM (
    'tiff',
    'geotiff',
    'multispectral',
    'rgb',
    'grayscale'
);

CREATE TYPE pipeline.api_source AS ENUM (
    'usgs_3dep',
    'open_meteo',
    'esa_worldcover',
    'nasa_srtm',
    'custom'
);

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create function for generating UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create audit log function
CREATE OR REPLACE FUNCTION pipeline.create_audit_log() 
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO pipeline.audit_log (
            table_name, 
            operation, 
            row_id, 
            new_values, 
            changed_by, 
            changed_at
        ) VALUES (
            TG_TABLE_NAME, 
            TG_OP, 
            NEW.id, 
            row_to_json(NEW), 
            current_user, 
            CURRENT_TIMESTAMP
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO pipeline.audit_log (
            table_name, 
            operation, 
            row_id, 
            old_values, 
            new_values, 
            changed_by, 
            changed_at
        ) VALUES (
            TG_TABLE_NAME, 
            TG_OP, 
            NEW.id, 
            row_to_json(OLD), 
            row_to_json(NEW), 
            current_user, 
            CURRENT_TIMESTAMP
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO pipeline.audit_log (
            table_name, 
            operation, 
            row_id, 
            old_values, 
            changed_by, 
            changed_at
        ) VALUES (
            TG_TABLE_NAME, 
            TG_OP, 
            OLD.id, 
            row_to_json(OLD), 
            current_user, 
            CURRENT_TIMESTAMP
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create utility functions for coordinate validation
CREATE OR REPLACE FUNCTION pipeline.is_valid_latitude(lat DECIMAL)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN lat >= -90 AND lat <= 90;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION pipeline.is_valid_longitude(lon DECIMAL)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN lon >= -180 AND lon <= 180;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to calculate distance between coordinates
CREATE OR REPLACE FUNCTION pipeline.calculate_distance_km(
    lat1 DECIMAL, 
    lon1 DECIMAL, 
    lat2 DECIMAL, 
    lon2 DECIMAL
)
RETURNS DECIMAL AS $$
BEGIN
    RETURN ST_Distance(
        ST_GeogFromText('POINT(' || lon1 || ' ' || lat1 || ')'),
        ST_GeogFromText('POINT(' || lon2 || ' ' || lat2 || ')')
    ) / 1000; -- Convert meters to kilometers
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Set timezone to UTC for consistency
SET timezone = 'UTC';

-- Add comments for documentation
COMMENT ON SCHEMA pipeline IS 'Main schema for TIFF enrichment pipeline tables';
COMMENT ON SCHEMA metadata IS 'Schema for file metadata and enrichment data';
COMMENT ON SCHEMA monitoring IS 'Schema for monitoring, metrics, and audit logs';

COMMENT ON TYPE pipeline.processing_status IS 'Status of file processing in the pipeline';
COMMENT ON TYPE pipeline.file_type IS 'Type of TIFF file being processed';
COMMENT ON TYPE pipeline.api_source IS 'External API sources used for enrichment data';

-- Create indices for performance (will be created with tables in next script)
-- But prepare some common index creation functions

CREATE OR REPLACE FUNCTION pipeline.create_common_indices(table_name TEXT)
RETURNS VOID AS $$
BEGIN
    -- Create index on created_at for time-based queries
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_' || table_name || '_created_at ON pipeline.' || table_name || ' (created_at)';
    
    -- Create index on updated_at for change tracking
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_' || table_name || '_updated_at ON pipeline.' || table_name || ' (updated_at)';
    
    -- Create index on status if the table has it
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'pipeline' 
        AND table_name = table_name 
        AND column_name = 'status'
    ) THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_' || table_name || '_status ON pipeline.' || table_name || ' (status)';
    END IF;
END;
$$ LANGUAGE plpgsql;