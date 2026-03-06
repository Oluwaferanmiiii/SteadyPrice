import React, { useState } from 'react';
import { ArrowPathIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { getApi, postApi } from '../services/api';
import toast from 'react-hot-toast';

const Prediction = () => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    category: 'Electronics',
    model_type: 'ensemble'
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const categories = [
    'Appliances',
    'Automotive', 
    'Electronics',
    'Office_Products',
    'Tools_and_Home_Improvement',
    'Cell_Phones_and_Accessories',
    'Toys_and_Games',
    'Musical_Instruments'
  ];

  const modelTypes = [
    { value: 'ensemble', label: 'Ensemble (Recommended)' },
    { value: 'traditional_ml', label: 'Traditional ML' },
    { value: 'deep_learning', label: 'Deep Learning' },
    { value: 'fine_tuned_llm', label: 'Fine-tuned LLM' }
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.title.trim()) {
      toast.error('Please enter a product title');
      return;
    }

    setLoading(true);
    setPrediction(null);

    try {
      const response = await postApi('/predictions/predict', formData);
      setPrediction(response.data);
      toast.success('Price prediction completed!');
    } catch (error) {
      toast.error('Failed to generate prediction');
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      title: '',
      description: '',
      category: 'Electronics',
      model_type: 'ensemble'
    });
    setPrediction(null);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Price Prediction</h1>
        <p className="text-gray-600 mt-2">Get AI-powered price predictions for your products</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction Form */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Product Information</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Product Title *
              </label>
              <input
                type="text"
                name="title"
                value={formData.title}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="e.g., Samsung 55-inch 4K Smart TV"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description
              </label>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Detailed product description..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Category
              </label>
              <select
                name="category"
                value={formData.category}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category.replace('_', ' ')}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Type
              </label>
              <select
                name="model_type"
                value={formData.model_type}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {modelTypes.map(model => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex space-x-3">
              <button
                type="submit"
                disabled={loading}
                className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </div>
                ) : (
                  'Predict Price'
                )}
              </button>
              <button
                type="button"
                onClick={resetForm}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <ArrowPathIcon className="h-5 w-5" />
              </button>
            </div>
          </form>
        </div>

        {/* Prediction Result */}
        <div className="space-y-6">
          {prediction ? (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center mb-4">
                <CheckCircleIcon className="h-6 w-6 text-green-500 mr-2" />
                <h2 className="text-lg font-semibold text-gray-900">Prediction Result</h2>
              </div>
              
              <div className="space-y-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-sm text-blue-600 mb-1">Predicted Price</p>
                  <p className="text-3xl font-bold text-blue-900">
                    ${prediction.predicted_price}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-xs text-gray-600 mb-1">Confidence</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {(prediction.confidence_score * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-xs text-gray-600 mb-1">Model Used</p>
                    <p className="text-sm font-medium text-gray-900 capitalize">
                      {prediction.model_used.replace('_', ' ')}
                    </p>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-600 mb-2">Price Range</p>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-700">Min: ${prediction.price_range.min.toFixed(2)}</span>
                    <span className="text-gray-700">Max: ${prediction.price_range.max.toFixed(2)}</span>
                  </div>
                </div>

                <div className="text-xs text-gray-500">
                  <p>Processing time: {prediction.processing_time_ms.toFixed(2)}ms</p>
                  <p>Generated: {new Date(prediction.timestamp).toLocaleString()}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-center py-8">
                <div className="bg-gray-100 rounded-full p-4 inline-block mb-4">
                  <ArrowPathIcon className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Prediction Yet</h3>
                <p className="text-gray-500 text-sm">
                  Fill in the product information and click "Predict Price" to get started
                </p>
              </div>
            </div>
          )}

          {/* Sample Products */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Try These Examples</h3>
            <div className="space-y-2">
              {[
                { title: 'Apple iPhone 15 Pro', category: 'Cell_Phones_and_Accessories' },
                { title: 'Dyson Vacuum Cleaner V15', category: 'Appliances' },
                { title: 'Sony PlayStation 5', category: 'Electronics' }
              ].map((example, index) => (
                <button
                  key={index}
                  onClick={() => setFormData(prev => ({ ...prev, ...example }))}
                  className="w-full text-left px-3 py-2 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <p className="text-sm font-medium text-gray-900">{example.title}</p>
                  <p className="text-xs text-gray-500">{example.category.replace('_', ' ')}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Prediction;
