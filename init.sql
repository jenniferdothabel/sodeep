-- DEEP ANAL Database Initialization Script
-- Creates tables for analysis result storage

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    entropy_value FLOAT NOT NULL,
    meta_data TEXT,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    thumbnail BYTEA,
    detection_likelihood FLOAT,
    detection_confidence VARCHAR(20),
    indicators JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_results(analysis_date);
CREATE INDEX IF NOT EXISTS idx_file_type ON analysis_results(file_type);
CREATE INDEX IF NOT EXISTS idx_detection_likelihood ON analysis_results(detection_likelihood);
CREATE INDEX IF NOT EXISTS idx_filename ON analysis_results(filename);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_analysis_results_updated_at
    BEFORE UPDATE ON analysis_results
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

-- Insert sample data for testing (optional)
INSERT INTO analysis_results (filename, file_size, file_type, entropy_value, meta_data, detection_likelihood, detection_confidence) 
VALUES 
('sample_clean.png', 15432, 'png', 7.2, '{"sample": "metadata"}', 0.15, 'Low'),
('sample_stego.png', 18765, 'png', 7.8, '{"sample": "metadata"}', 0.85, 'High')
ON CONFLICT DO NOTHING;