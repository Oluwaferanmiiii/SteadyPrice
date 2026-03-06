-- SteadyPrice Enterprise Database Schema
-- PostgreSQL initialization script

-- Create database (handled by PostgreSQL container)

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    features JSONB,
    weight DECIMAL(8,2),
    brand VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    product_title VARCHAR(500) NOT NULL,
    product_description TEXT,
    product_category VARCHAR(100) NOT NULL,
    predicted_price DECIMAL(10,2) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    price_range JSONB,
    model_used VARCHAR(50) NOT NULL,
    processing_time_ms DECIMAL(8,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    accuracy DECIMAL(3,2) NOT NULL,
    mae DECIMAL(8,2) NOT NULL,
    rmse DECIMAL(8,2) NOT NULL,
    mape DECIMAL(5,2) NOT NULL,
    r2_score DECIMAL(3,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_price ON products(price);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model_used ON predictions(model_used);

-- Insert demo users
INSERT INTO users (email, name, password_hash, role) VALUES
('admin@steadyprice.ai', 'Admin User', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6QJw/2Ej7W', 'admin'),
('user@steadyprice.ai', 'Demo User', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6QJw/2Ej7W', 'user')
ON CONFLICT (email) DO NOTHING;

-- Insert sample model metrics
INSERT INTO model_metrics (model_type, accuracy, mae, rmse, mape, r2_score) VALUES
('traditional_ml', 0.85, 12.50, 18.75, 14.20, 0.82),
('deep_learning', 0.87, 11.25, 16.88, 12.80, 0.84),
('fine_tuned_llm', 0.89, 10.00, 15.00, 11.40, 0.86),
('ensemble', 0.91, 8.75, 13.13, 9.95, 0.88)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
