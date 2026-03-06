import React, { useState, useEffect } from 'react';
import { CurrencyDollarIcon, ChartBarIcon, CpuChipIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { getApi } from '../services/api';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalPredictions: 0,
    averageAccuracy: 0,
    modelsLoaded: 0,
    systemHealth: 'healthy'
  });
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch health status
      const healthResponse = await getApi('/health');
      const healthData = healthResponse.data;
      
      // Fetch recent predictions (mock data for demo)
      const mockPredictions = [
        { id: 1, title: 'Samsung Smart TV', category: 'Electronics', price: 899.99, confidence: 0.92, time: '2 mins ago' },
        { id: 2, title: 'KitchenAid Mixer', category: 'Appliances', price: 349.99, confidence: 0.88, time: '5 mins ago' },
        { id: 3, title: 'Power Drill Set', category: 'Tools', price: 129.99, confidence: 0.85, time: '8 mins ago' },
        { id: 4, title: 'Wireless Headphones', category: 'Electronics', price: 199.99, confidence: 0.91, time: '12 mins ago' },
        { id: 5, title: 'Coffee Maker', category: 'Appliances', price: 79.99, confidence: 0.87, time: '15 mins ago' }
      ];

      setStats({
        totalPredictions: 1247,
        averageAccuracy: 0.89,
        modelsLoaded: healthData.models_loaded ? 3 : 0,
        systemHealth: healthData.status
      });
      
      setRecentPredictions(mockPredictions);
    } catch (error) {
      toast.error('Failed to fetch dashboard data');
      console.error('Dashboard error:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Predictions',
      value: stats.totalPredictions.toLocaleString(),
      icon: CurrencyDollarIcon,
      color: 'bg-blue-500',
      change: '+12% from last week'
    },
    {
      title: 'Average Accuracy',
      value: `${(stats.averageAccuracy * 100).toFixed(1)}%`,
      icon: ChartBarIcon,
      color: 'bg-green-500',
      change: '+3.2% from last week'
    },
    {
      title: 'Models Loaded',
      value: stats.modelsLoaded,
      icon: CpuChipIcon,
      color: 'bg-purple-500',
      change: 'All systems operational'
    },
    {
      title: 'System Health',
      value: stats.systemHealth,
      icon: CheckCircleIcon,
      color: stats.systemHealth === 'healthy' ? 'bg-green-500' : 'bg-red-500',
      change: 'Last checked: just now'
    }
  ];

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">SteadyPrice Dashboard</h1>
        <p className="text-gray-600 mt-2">Enterprise Price Prediction Platform</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <div key={index} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">{stat.title}</p>
                <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
                <p className="text-xs text-gray-500 mt-1">{stat.change}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Predictions */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Predictions</h2>
        </div>
        <div className="divide-y divide-gray-200">
          {recentPredictions.map((prediction) => (
            <div key={prediction.id} className="px-6 py-4 hover:bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-gray-900">{prediction.title}</h3>
                  <div className="flex items-center mt-1 space-x-4">
                    <span className="text-xs text-gray-500">{prediction.category}</span>
                    <span className="text-xs text-gray-500">{prediction.time}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <p className="text-lg font-semibold text-gray-900">${prediction.price}</p>
                    <p className="text-xs text-gray-500">
                      {(prediction.confidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>
                  <div className={`w-2 h-2 rounded-full ${
                    prediction.confidence > 0.9 ? 'bg-green-500' : 
                    prediction.confidence > 0.8 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            New Price Prediction
          </button>
          <button className="px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
            View Analytics
          </button>
          <button className="px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
            Model Performance
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
