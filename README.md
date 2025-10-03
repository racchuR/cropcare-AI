# CropCare AI 🌱

**AI-powered crop disease detection and treatment advice for farmers**

CropCare AI is a comprehensive web application that uses machine learning to detect crop diseases from images and provides intelligent treatment recommendations. Built with modern technologies, it offers farmers an accessible tool for early disease detection and management.

## 🚀 Features

### Core Functionality
- **AI Disease Detection**: Upload crop images and get instant disease identification with confidence scores
- **Treatment Recommendations**: Receive AI-powered treatment advice based on detected diseases
- **User Authentication**: Secure JWT-based authentication system
- **Prediction History**: Track and manage your disease detection history
- **Dashboard Analytics**: Visualize your crop health data and trends

### Technical Features
- **Multi-language Support**: Ready for localization (English + local languages)
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Real-time Processing**: Fast image analysis and results
- **Secure File Upload**: Safe image handling with validation
- **API Integration**: OpenAI API for enhanced treatment advice

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **PostgreSQL**: Robust relational database for data persistence
- **SQLAlchemy**: Python SQL toolkit and ORM
- **Alembic**: Database migration tool
- **JWT**: JSON Web Tokens for authentication
- **PyTorch**: Machine learning framework for disease detection

### Frontend
- **React 18**: Modern JavaScript library for building user interfaces
- **Tailwind CSS**: Utility-first CSS framework for styling
- **React Router**: Client-side routing
- **Axios**: HTTP client for API communication
- **Lucide React**: Beautiful icon library

### AI/ML
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **OpenCV**: Image processing library
- **OpenAI API**: GPT integration for treatment advice

### Infrastructure
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container application orchestration
- **Nginx**: Reverse proxy and web server
- **Redis**: Caching and session storage (optional)

## 📁 Project Structure

```
CropCare-AI/
├── backend/                 # FastAPI backend application
│   ├── main.py             # Main application entry point
│   ├── config.py           # Configuration settings
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile         # Backend container configuration
│   ├── database/          # Database models and configuration
│   ├── ml_model/          # ML inference module
│   └── tests/             # Backend test suite
├── frontend/               # React frontend application
│   ├── src/               # Source code
│   │   ├── components/    # Reusable React components
│   │   ├── pages/         # Page components
│   │   ├── contexts/      # React context providers
│   │   └── utils/         # Utility functions
│   ├── public/            # Static assets
│   ├── package.json       # Node.js dependencies
│   ├── Dockerfile        # Frontend container configuration
│   └── nginx.conf        # Nginx configuration
├── ml-model/              # Machine learning model training
│   ├── train.py          # Model training script
│   ├── inference.py      # Standalone inference script
│   ├── data_preparation.py # Dataset preparation utilities
│   └── requirements.txt  # ML dependencies
├── database/              # Database schema and migrations
│   ├── init.sql          # Database initialization
│   ├── alembic.ini       # Alembic configuration
│   └── migrations/       # Database migration files
├── nginx/                 # Nginx configuration
├── docker-compose.yml     # Docker Compose configuration
├── env.example           # Environment variables template
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Git
- OpenAI API key (optional, for enhanced treatment advice)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CropCare-AI.git
cd CropCare-AI
```

### 2. Environment Setup
```bash
# Copy environment template
cp env.example .env

# Edit .env file with your configuration
# At minimum, update:
# - SECRET_KEY (generate a secure random key)
# - OPENAI_API_KEY (optional, for enhanced features)
```

### 3. Start the Application
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 5. Initial Setup
1. Register a new account at http://localhost:3000/register
2. Login to access the dashboard
3. Upload your first crop image for disease detection

## 🔧 Development Setup

### Backend Development
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### ML Model Training
```bash
cd ml-model

# Install dependencies
pip install -r requirements.txt

# Prepare dataset
python data_preparation.py --action organize --source_dir /path/to/images --dataset_dir data

# Train model
python train.py --data_dir data --epochs 50 --batch_size 32
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Linting
```bash
# Backend
cd backend
flake8 .
black .

# Frontend
cd frontend
npm run lint
npm run format
```

## 📊 API Documentation

### Authentication Endpoints
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user info

### Prediction Endpoints
- `POST /predict` - Upload image for disease detection
- `GET /predictions` - Get user's prediction history

### Health Check
- `GET /health` - API health status

For detailed API documentation, visit http://localhost:8000/docs when the backend is running.

## 🌍 Disease Detection

The AI model can detect the following crop diseases:
- **Healthy** - No disease detected
- **Bacterial Blight** - Bacterial infection
- **Brown Spot** - Fungal disease
- **Leaf Blight** - Leaf infection
- **Leaf Scald** - Leaf damage
- **Leaf Spot** - Spot disease
- **Rust** - Rust infection
- **Smut** - Smut disease

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- CORS protection
- File upload validation
- SQL injection prevention
- XSS protection
- Rate limiting (via Nginx)

## 🚀 Deployment

### Production Deployment
1. Update environment variables for production
2. Configure SSL certificates
3. Set up domain and DNS
4. Deploy using Docker Compose or Kubernetes

### Environment Variables
Key environment variables for production:
```bash
SECRET_KEY=your-secure-secret-key
DATABASE_URL=postgresql://user:password@host:port/database
OPENAI_API_KEY=your-openai-api-key
DEBUG=false
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/React code
- Write tests for new features
- Update documentation as needed

## 📈 Roadmap

### Phase 1 (Current)
- ✅ Basic disease detection
- ✅ User authentication
- ✅ Web interface
- ✅ Docker deployment

### Phase 2 (Next)
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Offline mode
- [ ] Batch image processing

### Phase 3 (Future)
- [ ] IoT sensor integration
- [ ] Weather data integration
- [ ] Marketplace for treatments
- [ ] Community features
- [ ] Advanced ML models

## 🐛 Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart db
```

**Frontend Build Error**
```bash
# Clear node modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**ML Model Not Loading**
```bash
# Check if model file exists
ls -la ml-model/saved_models/

# Train a new model if needed
cd ml-model
python train.py --data_dir data
```

## 📞 Support

- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Email**: support@cropcare-ai.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** - For the excellent ML framework
- **FastAPI Team** - For the modern Python web framework
- **React Team** - For the powerful frontend library
- **OpenAI** - For the GPT API integration
- **Contributors** - Thank you to all contributors who help improve this project

## 📊 Project Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-80%25-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

---

**Made with ❤️ for farmers worldwide**

*CropCare AI - Empowering farmers with AI-driven crop health solutions*
