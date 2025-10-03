-- CropCare AI Database Schema
-- PostgreSQL initialization script

-- Create database (run this as superuser)
-- CREATE DATABASE cropcare_db;
-- CREATE USER cropcare WITH PASSWORD 'password';
-- GRANT ALL PRIVILEGES ON DATABASE cropcare_db TO cropcare;

-- Connect to the database
\c cropcare_db;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create crop_images table
CREATE TABLE IF NOT EXISTS crop_images (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create disease_predictions table
CREATE TABLE IF NOT EXISTS disease_predictions (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES crop_images(id) ON DELETE CASCADE,
    disease_name VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    treatment_advice TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_crop_images_user_id ON crop_images(user_id);
CREATE INDEX IF NOT EXISTS idx_crop_images_created_at ON crop_images(created_at);
CREATE INDEX IF NOT EXISTS idx_disease_predictions_image_id ON disease_predictions(image_id);
CREATE INDEX IF NOT EXISTS idx_disease_predictions_disease_name ON disease_predictions(disease_name);
CREATE INDEX IF NOT EXISTS idx_disease_predictions_created_at ON disease_predictions(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for development
INSERT INTO users (username, email, hashed_password, full_name) VALUES
('demo_user', 'demo@cropcare.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBdX8X8X8X8X8X', 'Demo User'),
('farmer_john', 'john@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBdX8X8X8X8X', 'John Farmer')
ON CONFLICT (username) DO NOTHING;

-- Create view for user predictions with image info
CREATE OR REPLACE VIEW user_predictions AS
SELECT 
    dp.id as prediction_id,
    dp.disease_name,
    dp.confidence_score,
    dp.treatment_advice,
    dp.created_at as prediction_date,
    ci.id as image_id,
    ci.original_filename,
    ci.file_path,
    u.id as user_id,
    u.username,
    u.full_name
FROM disease_predictions dp
JOIN crop_images ci ON dp.image_id = ci.id
JOIN users u ON ci.user_id = u.id
ORDER BY dp.created_at DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cropcare;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cropcare;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO cropcare;
