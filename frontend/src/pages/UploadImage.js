import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import toast from 'react-hot-toast';
import { 
  Upload, 
  Camera, 
  CheckCircle, 
  AlertTriangle, 
  X,
  Loader,
  Leaf
} from 'lucide-react';

const UploadImage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const removeFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
      toast.success('Image analyzed successfully!');
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to analyze image. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const getDiseaseStatusColor = (disease) => {
    if (disease === 'Healthy') return 'text-green-600 bg-green-100';
    if (disease.includes('Blight') || disease.includes('Spot')) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getDiseaseIcon = (disease) => {
    if (disease === 'Healthy') return <CheckCircle className="w-5 h-5" />;
    return <AlertTriangle className="w-5 h-5" />;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Crop Image</h1>
        <p className="text-gray-600">
          Upload an image of your crop to get AI-powered disease detection and treatment advice
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Select Image</h2>
            
            {!selectedFile ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
                  isDragActive
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                {isDragActive ? (
                  <p className="text-primary-600 font-medium">Drop the image here...</p>
                ) : (
                  <div>
                    <p className="text-gray-600 mb-2">
                      Drag & drop an image here, or click to select
                    </p>
                    <p className="text-sm text-gray-500">
                      Supports JPG, PNG, BMP up to 10MB
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="relative">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-64 object-cover rounded-lg"
                />
                <button
                  onClick={removeFile}
                  className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors duration-200"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {selectedFile && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{selectedFile.name}</p>
                    <p className="text-sm text-gray-600">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {uploading ? (
                      <div className="flex items-center space-x-2">
                        <Loader className="w-4 h-4 animate-spin" />
                        <span>Analyzing...</span>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <Camera className="w-4 h-4" />
                        <span>Analyze Image</span>
                      </div>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Tips for Best Results</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <span>Use clear, well-lit images of leaves or affected areas</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <span>Ensure the crop is clearly visible and in focus</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <span>Avoid images with multiple crops or backgrounds</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                <span>Take photos from different angles if symptoms vary</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          {prediction ? (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h2>
              
              <div className="space-y-6">
                {/* Disease Detection */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`p-2 rounded-lg ${getDiseaseStatusColor(prediction.disease)}`}>
                      {getDiseaseIcon(prediction.disease)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{prediction.disease}</h3>
                      <p className="text-sm text-gray-600">
                        Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  
                  {/* Confidence Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        prediction.confidence > 0.8
                          ? 'bg-green-500'
                          : prediction.confidence > 0.6
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Treatment Advice */}
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-2 flex items-center space-x-2">
                    <Leaf className="w-4 h-4 text-blue-600" />
                    <span>Treatment Advice</span>
                  </h3>
                  <p className="text-sm text-gray-700 leading-relaxed">
                    {prediction.treatment_advice}
                  </p>
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-3">
                  <button
                    onClick={() => {
                      setSelectedFile(null);
                      setPreview(null);
                      setPrediction(null);
                    }}
                    className="btn-secondary flex-1"
                  >
                    Analyze Another Image
                  </button>
                  <button
                    onClick={() => window.print()}
                    className="btn-primary flex-1"
                  >
                    Save Report
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="text-center py-12">
                <Camera className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  No Analysis Yet
                </h3>
                <p className="text-gray-600">
                  Upload and analyze an image to see results here
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadImage;
