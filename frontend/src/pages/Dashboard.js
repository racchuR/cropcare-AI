import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { 
  Upload, 
  History, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  Leaf,
  Camera,
  BarChart3
} from 'lucide-react';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalPredictions: 0,
    healthyPlants: 0,
    diseasedPlants: 0,
    recentPredictions: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get('/predictions');
      const predictions = response.data;
      
      const totalPredictions = predictions.length;
      const healthyPlants = predictions.filter(p => p.disease_name === 'Healthy').length;
      const diseasedPlants = totalPredictions - healthyPlants;
      const recentPredictions = predictions.slice(0, 5);

      setStats({
        totalPredictions,
        healthyPlants,
        diseasedPlants,
        recentPredictions
      });
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDiseaseStatusColor = (disease) => {
    if (disease === 'Healthy') return 'text-green-600 bg-green-100';
    if (disease.includes('Blight') || disease.includes('Spot')) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getDiseaseIcon = (disease) => {
    if (disease === 'Healthy') return <CheckCircle className="w-4 h-4" />;
    return <AlertTriangle className="w-4 h-4" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="flex items-center space-x-2">
          <div className="spinner"></div>
          <span>Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2">
          Welcome back, {user?.full_name}!
        </h1>
        <p className="text-primary-100">
          Monitor your crops and get AI-powered disease detection insights.
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          to="/upload"
          className="card hover:shadow-md transition-shadow duration-200 group"
        >
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center group-hover:bg-primary-200 transition-colors duration-200">
              <Camera className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Upload New Image</h3>
              <p className="text-gray-600">Analyze crop images for disease detection</p>
            </div>
          </div>
        </Link>

        <Link
          to="/history"
          className="card hover:shadow-md transition-shadow duration-200 group"
        >
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors duration-200">
              <BarChart3 className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">View History</h3>
              <p className="text-gray-600">Check your previous predictions and reports</p>
            </div>
          </div>
        </Link>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Predictions</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalPredictions}</p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Healthy Plants</p>
              <p className="text-2xl font-bold text-green-600">{stats.healthyPlants}</p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Diseased Plants</p>
              <p className="text-2xl font-bold text-red-600">{stats.diseasedPlants}</p>
            </div>
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">Recent Predictions</h2>
          <Link
            to="/history"
            className="text-primary-600 hover:text-primary-700 font-medium transition-colors duration-200"
          >
            View all
          </Link>
        </div>

        {stats.recentPredictions.length === 0 ? (
          <div className="text-center py-12">
            <Leaf className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No predictions yet</h3>
            <p className="text-gray-600 mb-6">
              Upload your first crop image to get started with disease detection.
            </p>
            <Link
              to="/upload"
              className="btn-primary inline-flex items-center space-x-2"
            >
              <Camera className="w-4 h-4" />
              <span>Upload Image</span>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {stats.recentPredictions.map((prediction) => (
              <div
                key={prediction.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-4">
                  <div className={`p-2 rounded-lg ${getDiseaseStatusColor(prediction.disease_name)}`}>
                    {getDiseaseIcon(prediction.disease_name)}
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{prediction.disease_name}</p>
                    <p className="text-sm text-gray-600">
                      Confidence: {(prediction.confidence_score * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">
                    {new Date(prediction.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
