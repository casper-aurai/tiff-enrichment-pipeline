-- =============================================================================
-- TIFF Enrichment Pipeline - Table Creation
-- This script creates all tables needed for the pipeline operation
-- =============================================================================

-- Set search path
SET search_path TO pipeline, metadata, monitoring, public;

-- =============================================================================
-- MAIN PIPELINE TABLES
-- =============================================================================

-- Processing runs table - tracks each file processing session
CREATE TABLE IF NOT EXISTS pipeline.processing_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_file_path VARCHAR(512) NOT NULL,
    input_file_name VARCHAR(255) NOT NULL,
    input_file_size_bytes BIGINT,
    input_file_checksum VARCHAR(64), -- SHA-256 hash
    
    -- File metadata
    file_type pipeline.file_type,
    image_width INTEGER,
    image_height INTEGER,
    pixel_depth INTEGER,
    color_space VARCHAR(50),
    
    -- Processing information
    status pipeline.processing_status NOT NULL DEFAULT 'pending',
    processing_start_time TIMESTAMP WITH TIME ZONE,
    processing_end_time TIMESTAMP WITH TIME ZONE,
    processing_duration_seconds INTEGER GENERATED ALWAYS AS (
        CASE 
            WHEN processing_end_time IS NOT NULL AND processing_start_time IS NOT NULL 
            THEN EXTRACT(EPOCH FROM (processing_end_time - processing_start_time))::INTEGER
            ELSE NULL 
        END
    ) STORED,
    
    -- Output information
    output_tiff_path VARCHAR(512),
    output_json_path VARCHAR(512),
    output_file_size_bytes BIGINT,
    output_file_checksum VARCHAR(64),
    
    -- Error handling
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    error_details JSONB,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT current_user,
    
    -- Constraints
    CONSTRAINT chk_processing_times CHECK (
        processing_end_time IS NULL OR processing_start_time IS NULL OR 
        processing_end_time >= processing_start_time
    ),
    CONSTRAINT chk_file_size CHECK (input_file_size_bytes > 0),
    CONSTRAINT chk_retry_count CHECK (retry_count >= 0)
);

-- Geospatial metadata table
CREATE TABLE IF NOT EXISTS metadata.geospatial_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    processing_run_id UUID NOT NULL REFERENCES pipeline.processing_runs(id) ON DELETE CASCADE,
    
    -- Original coordinates from EXIF/GeoTIFF
    original_latitude DECIMAL(10, 8),
    original_longitude DECIMAL(11, 8),
    original_coordinate_system VARCHAR(50),
    original_geom GEOMETRY(POINT, 4326), -- PostGIS geometry
    
    -- Standardized coordinates (WGS84)
    wgs84_latitude DECIMAL(10, 8) NOT NULL,
    wgs84_longitude DECIMAL(11, 8) NOT NULL,
    wgs84_geom GEOMETRY(POINT, 4326) NOT NULL,
    
    -- EXIF metadata
    capture_timestamp TIMESTAMP WITH TIME ZONE,
    camera_make VARCHAR(100),
    camera_model VARCHAR(100),
    lens_model VARCHAR(100),
    focal_length_mm DECIMAL(5, 2),
    aperture_f_stop DECIMAL(3, 1),
    shutter_speed VARCHAR(20),
    iso_speed INTEGER,
    
    -- Image technical metadata
    pixel_x_dimension INTEGER,
    pixel_y_dimension INTEGER,
    resolution_x DECIMAL(8, 4),
    resolution_y DECIMAL(8, 4),
    resolution_unit VARCHAR(20),
    color_space VARCHAR(50),
    compression_type VARCHAR(50),
    
    -- Validation
    metadata_completeness_score DECIMAL(3, 2) DEFAULT 0.0, -- 0.0 to 1.0
    coordinate_precision_meters DECIMAL(8, 2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_latitude CHECK (pipeline.is_valid_latitude(wgs84_latitude)),
    CONSTRAINT chk_longitude CHECK (pipeline.is_valid_longitude(wgs84_longitude)),
    CONSTRAINT chk_completeness_score CHECK (
        metadata_completeness_score >= 0.0 AND metadata_completeness_score <= 1.0
    ),
    CONSTRAINT chk_image_dimensions CHECK (
        pixel_x_dimension > 0 AND pixel_y_dimension > 0
    )
);

-- Enrichment data table - stores data fetched from external APIs
CREATE TABLE IF NOT EXISTS metadata.enrichment_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    processing_run_id UUID NOT NULL REFERENCES pipeline.processing_runs(id) ON DELETE CASCADE,
    
    -- Elevation data
    elevation_meters DECIMAL(8, 2),
    elevation_source pipeline.api_source,
    elevation_accuracy_meters DECIMAL(6, 2),
    elevation_fetch_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Weather data
    weather_data JSONB, -- Flexible storage for weather information
    weather_source pipeline.api_source,
    weather_fetch_timestamp TIMESTAMP WITH TIME ZONE,
    weather_historical_date DATE, -- Date for which historical weather was fetched
    
    -- Land cover data
    land_cover_class VARCHAR(100),
    land_cover_confidence DECIMAL(3, 2), -- 0.0 to 1.0
    land_cover_source pipeline.api_source,
    land_cover_fetch_timestamp TIMESTAMP WITH TIME ZONE,
    land_cover_year INTEGER,
    
    -- Additional enrichment data (flexible JSONB storage)
    additional_data JSONB,
    
    -- API performance metrics
    total_api_calls INTEGER DEFAULT 0,
    total_api_response_time_ms INTEGER DEFAULT 0,
    api_error_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_land_cover_confidence CHECK (
        land_cover_confidence IS NULL OR 
        (land_cover_confidence >= 0.0 AND land_cover_confidence <= 1.0)
    ),
    CONSTRAINT chk_api_metrics CHECK (
        total_api_calls >= 0 AND 
        total_api_response_time_ms >= 0 AND 
        api_error_count >= 0
    )
);

-- =============================================================================
-- MONITORING AND AUDIT TABLES
-- =============================================================================

-- Audit log for tracking all changes
CREATE TABLE IF NOT EXISTS pipeline.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(10) NOT NULL, -- INSERT, UPDATE, DELETE
    row_id UUID,
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_operation CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE'))
);

-- System metrics for monitoring pipeline performance
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 4) NOT NULL,
    metric_unit VARCHAR(20),
    metric_tags JSONB, -- For additional dimensions
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite index for efficient time-series queries
    UNIQUE(metric_name, recorded_at)
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS monitoring.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_source pipeline.api_source NOT NULL,
    endpoint_url VARCHAR(512),
    request_method VARCHAR(10) DEFAULT 'GET',
    
    -- Request details
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    response_status_code INTEGER,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    
    -- Success/failure tracking
    is_successful BOOLEAN GENERATED ALWAYS AS (
        response_status_code BETWEEN 200 AND 299
    ) STORED,
    error_message TEXT,
    
    -- Rate limiting and quotas
    rate_limit_remaining INTEGER,
    rate_limit_reset_time TIMESTAMP WITH TIME ZONE,
    
    -- Caching information
    cache_hit BOOLEAN DEFAULT FALSE,
    cache_key VARCHAR(255),
    
    processing_run_id UUID REFERENCES pipeline.processing_runs(id) ON DELETE SET NULL
);

-- Pipeline configuration table
CREATE TABLE IF NOT EXISTS pipeline.configuration (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) NOT NULL, -- 'string', 'number', 'boolean', 'object', 'array'
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE, -- For passwords, API keys, etc.
    
    -- Version control
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT current_user,
    
    CONSTRAINT chk_config_type CHECK (
        config_type IN ('string', 'number', 'boolean', 'object', 'array')
    )
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Processing runs indexes
CREATE INDEX IF NOT EXISTS idx_processing_runs_status ON pipeline.processing_runs (status);
CREATE INDEX IF NOT EXISTS idx_processing_runs_created_at ON pipeline.processing_runs (created_at);
CREATE INDEX IF NOT EXISTS idx_processing_runs_file_path ON pipeline.processing_runs (input_file_path);
CREATE INDEX IF NOT EXISTS idx_processing_runs_checksum ON pipeline.processing_runs (input_file_checksum);

-- Geospatial data indexes
CREATE INDEX IF NOT EXISTS idx_geospatial_data_coordinates ON metadata.geospatial_data (wgs84_latitude, wgs84_longitude);
CREATE INDEX IF NOT EXISTS idx_geospatial_data_geom ON metadata.geospatial_data USING GIST (wgs84_geom);
CREATE INDEX IF NOT EXISTS idx_geospatial_data_timestamp ON metadata.geospatial_data (capture_timestamp);

-- Enrichment data indexes
CREATE INDEX IF NOT EXISTS idx_enrichment_data_elevation ON metadata.enrichment_data (elevation_meters);
CREATE INDEX IF NOT EXISTS idx_enrichment_data_land_cover ON metadata.enrichment_data (land_cover_class);
CREATE INDEX IF NOT EXISTS idx_enrichment_data_weather ON metadata.enrichment_data USING GIN (weather_data);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_table_operation ON pipeline.audit_log (table_name, operation);
CREATE INDEX IF NOT EXISTS idx_audit_log_changed_at ON pipeline.audit_log (changed_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON monitoring.system_metrics (metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_source_timestamp ON monitoring.api_usage (api_source, request_timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_successful ON monitoring.api_usage (is_successful);

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Update timestamps automatically
CREATE TRIGGER update_processing_runs_updated_at
    BEFORE UPDATE ON pipeline.processing_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configuration_updated_at
    BEFORE UPDATE ON pipeline.configuration
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit logging triggers
CREATE TRIGGER audit_processing_runs
    AFTER INSERT OR UPDATE OR DELETE ON pipeline.processing_runs
    FOR EACH ROW EXECUTE FUNCTION pipeline.create_audit_log();

CREATE TRIGGER audit_geospatial_data
    AFTER INSERT OR UPDATE OR DELETE ON metadata.geospatial_data
    FOR EACH ROW EXECUTE FUNCTION pipeline.create_audit_log();

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Complete processing status view
CREATE OR REPLACE VIEW pipeline.processing_status_summary AS
SELECT 
    pr.id,
    pr.input_file_name,
    pr.status,
    pr.processing_start_time,
    pr.processing_end_time,
    pr.processing_duration_seconds,
    pr.retry_count,
    pr.error_message,
    gd.wgs84_latitude,
    gd.wgs84_longitude,
    gd.capture_timestamp,
    ed.elevation_meters,
    ed.land_cover_class,
    pr.created_at
FROM pipeline.processing_runs pr
LEFT JOIN metadata.geospatial_data gd ON pr.id = gd.processing_run_id
LEFT JOIN metadata.enrichment_data ed ON pr.id = ed.processing_run_id
ORDER BY pr.created_at DESC;

-- API performance summary
CREATE OR REPLACE VIEW monitoring.api_performance_summary AS
SELECT 
    api_source,
    DATE_TRUNC('hour', request_timestamp) AS hour,
    COUNT(*) AS total_requests,
    COUNT(*) FILTER (WHERE is_successful) AS successful_requests,
    COUNT(*) FILTER (WHERE NOT is_successful) AS failed_requests,
    AVG(response_time_ms) AS avg_response_time_ms,
    MAX(response_time_ms) AS max_response_time_ms,
    COUNT(*) FILTER (WHERE cache_hit) AS cache_hits
FROM monitoring.api_usage
WHERE request_timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY api_source, DATE_TRUNC('hour', request_timestamp)
ORDER BY hour DESC, api_source;

-- Daily processing statistics
CREATE OR REPLACE VIEW monitoring.daily_processing_stats AS
SELECT 
    DATE(created_at) AS processing_date,
    COUNT(*) AS total_files,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_files,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_files,
    COUNT(*) FILTER (WHERE status = 'pending') AS pending_files,
    AVG(processing_duration_seconds) AS avg_processing_time_seconds,
    SUM(input_file_size_bytes) AS total_input_size_bytes,
    SUM(output_file_size_bytes) AS total_output_size_bytes
FROM pipeline.processing_runs
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY processing_date DESC;

-- =============================================================================
-- PERMISSIONS
-- =============================================================================

-- Grant permissions to readonly role
GRANT SELECT ON ALL TABLES IN SCHEMA pipeline TO pipeline_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA metadata TO pipeline_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO pipeline_readonly;

-- Grant permissions to processor role
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pipeline TO pipeline_processor;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA metadata TO pipeline_processor;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA monitoring TO pipeline_processor;

-- Grant sequence usage
GRANT USAGE ON ALL SEQUENCES IN SCHEMA pipeline TO pipeline_processor;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA metadata TO pipeline_processor;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA monitoring TO pipeline_processor;

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default configuration values
INSERT INTO pipeline.configuration (config_key, config_value, config_type, description) VALUES
('max_retry_attempts', '3', 'number', 'Maximum number of retry attempts for failed processing'),
('api_timeout_seconds', '30', 'number', 'Timeout for external API calls'),
('batch_size', '5', 'number', 'Number of files to process in parallel'),
('supported_extensions', '["tif", "tiff", "TIF", "TIFF"]', 'array', 'Supported file extensions'),
('require_gps_coordinates', 'true', 'boolean', 'Whether GPS coordinates are required'),
('default_coordinate_system', '"EPSG:4326"', 'string', 'Default coordinate reference system')
ON CONFLICT (config_key) DO NOTHING;

-- Create default metric entries
INSERT INTO monitoring.system_metrics (metric_name, metric_value, metric_unit) VALUES
('pipeline_startup', 1, 'count')
ON CONFLICT DO NOTHING;