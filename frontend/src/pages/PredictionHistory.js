import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Calendar, 
  Eye, 
  Download, 
  Filter,
  Search,
  CheckCircle,
  AlertTriangle,
  Leaf
} from 'lucide-react';

const PredictionHistory = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDisease, setFilterDisease] = useState('all');
  const [selectedPrediction, setSelectedPrediction] = useState(null);

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const response = await axios.get('/predictions');
      setPredictions(response.data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
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

  const filteredPredictions = predictions.filter(prediction => {
    const matchesSearch = prediction.disease_name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterDisease === 'all' || prediction.disease_name === filterDisease;
    return matchesSearch && matchesFilter;
  });

  const uniqueDiseases = [...new Set(predictions.map(p => p.disease_name))];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="flex items-center space-x-2">
          <div className="spinner"></div>
          <span>Loading prediction history...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Prediction History</h1>
        <p className="text-gray-600">
          View and manage your crop disease detection history
        </p>
      </div>

      {/* Filters and Search */}
      <div className="card">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search by disease name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input-field pl-10"
              />
            </div>
          </div>
          <div className="sm:w-48">
            <select
              value={filterDisease}
              onChange={(e) => setFilterDisease(e.target.value)}
              className="input-field"
            >
              <option value="all">All Diseases</option>
              {uniqueDiseases.map(disease => (
                <option key={disease} value={disease}>{disease}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Predictions Grid */}
      {filteredPredictions.length === 0 ? (
        <div className="card text-center py-12">
          <Leaf className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No predictions found
          </h3>
          <p className="text-gray-600">
            {predictions.length === 0 
              ? "You haven't made any predictions yet. Upload an image to get started!"
              : "No predictions match your current filters."
            }
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPredictions.map((prediction) => (
            <div key={prediction.id} className="card hover:shadow-md transition-shadow duration-200">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getDiseaseStatusColor(prediction.disease_name)}`}>
                    {getDiseaseIcon(prediction.disease_name)}
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">{prediction.disease_name}</h3>
                    <p className="text-sm text-gray-600">
                      {(prediction.confidence_score * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    prediction.confidence_score > 0.8
                      ? 'bg-green-500'
                      : prediction.confidence_score > 0.6
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${prediction.confidence_score * 100}%` }}
                ></div>
              </div>

              <div className="flex items-center justify-between text-sm text-gray-600 mb-4">
                <div className="flex items-center space-x-1">
                  <Calendar className="w-4 h-4" />
                  <span>{new Date(prediction.created_at).toLocaleDateString()}</span>
                </div>
                <span>{new Date(prediction.created_at).toLocaleTimeString()}</span>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => setSelectedPrediction(prediction)}
                  className="btn-secondary flex-1 flex items-center justify-center space-x-2"
                >
                  <Eye className="w-4 h-4" />
                  <span>View Details</span>
                </button>
                <button className="btn-primary flex items-center justify-center px-3">
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Prediction Details Modal */}
      {selectedPrediction && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Prediction Details</h2>
                <button
                  onClick={() => setSelectedPrediction(null)}
                  className="text-gray-400 hover:text-gray-600 transition-colors duration-200"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="space-y-6">
                {/* Disease Information */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`p-2 rounded-lg ${getDiseaseStatusColor(selectedPrediction.disease_name)}`}>
                      {getDiseaseIcon(selectedPrediction.disease_name)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{selectedPrediction.disease_name}</h3>
                      <p className="text-sm text-gray-600">
                        Confidence: {(selectedPrediction.confidence_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 ${
                        selectedPrediction.confidence_score > 0.8
                          ? 'bg-green-500'
                          : selectedPrediction.confidence_score > 0.6
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${selectedPrediction.confidence_score * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Treatment Advice */}
                {selectedPrediction.treatment_advice && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h3 className="font-semibold text-gray-900 mb-2 flex items-center space-x-2">
                      <Leaf className="w-4 h-4 text-blue-600" />
                      <span>Treatment Advice</span>
                    </h3>
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {selectedPrediction.treatment_advice}
                    </p>
                  </div>
                )}

                {/* Metadata */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-3">Analysis Information</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Date:</span>
                      <p className="font-medium">
                        {new Date(selectedPrediction.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-600">Time:</span>
                      <p className="font-medium">
                        {new Date(selectedPrediction.created_at).toLocaleTimeString()}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-600">Image ID:</span>
                      <p className="font-medium">{selectedPrediction.image_id}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Prediction ID:</span>
                      <p className="font-medium">{selectedPrediction.id}</p>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex space-x-3">
                  <button className="btn-secondary flex-1">
                    Download Report
                  </button>
                  <button
                    onClick={() => setSelectedPrediction(null)}
                    className="btn-primary flex-1"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionHistory;
