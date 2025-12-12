# Vercel Deployment Guide

This Next.js application has been configured for deployment on Vercel with Python serverless functions.

## Architecture

- **Frontend**: Next.js React application with TypeScript
- **Backend**: Python serverless functions for model computation
- **Model**: AI progress modeling using numpy and scipy

## Files Structure

```
nextjs-progress/
├── api/
│   ├── compute.py           # Model computation serverless function
│   └── parameter-config.py  # Parameter configuration function
├── app/                     # Next.js app directory
├── components/              # React components
├── progress_model.py        # Core Python model (copied from parent)
├── model_config.py          # Model configuration (copied from parent)
├── input_data.csv           # Time series data (copied from parent)
├── requirements.txt         # Python dependencies
├── vercel.json             # Vercel configuration
└── package.json            # Node.js dependencies
```

## Deployment Steps

### 1. Prerequisites
- Vercel account
- Git repository with the code

### 2. Dependencies

**Python dependencies** (requirements.txt):
```
numpy>=1.21.0
scipy>=1.7.0
```

**Node.js dependencies** (package.json):
```json
{
  "dependencies": {
    "next": "15.5.2",
    "react": "19.1.0",
    "react-dom": "19.1.0",
    "recharts": "^3.1.2"
  }
}
```

### 3. Vercel Configuration

The `vercel.json` file configures Python runtime for serverless functions:

```json
{
  "functions": {
    "api/compute.py": {
      "runtime": "python3.9"
    },
    "api/parameter-config.py": {
      "runtime": "python3.9"
    }
  },
  "build": {
    "env": {
      "PYTHONPATH": "."
    }
  }
}
```

### 4. Deployment Commands

1. **Connect to Vercel**:
   ```bash
   npm install -g vercel
   vercel login
   ```

2. **Deploy**:
   ```bash
   vercel --prod
   ```

### 5. Environment Variables

No environment variables are required for basic functionality. The application uses:
- Local file paths for data (input_data.csv)
- Hardcoded model configuration (model_config.py)

## API Endpoints

### `/api/parameter-config` (GET)
Returns model parameter configuration including bounds and defaults.

**Response**:
```json
{
  "success": true,
  "bounds": { ... },
  "defaults": { ... },
  "metadata": { ... }
}
```

### `/api/compute` (POST)
Computes model trajectory with given parameters.

**Request**:
```json
{
  "parameters": {
    "present_doubling_time": 0.5,
    "doubling_difficulty_growth_factor": 0.92,
    "ac_time_horizon_minutes": 5.1
  },
  "time_range": [2025, 2030],
  "initial_progress": 0.0
}
```

**Response**:
```json
{
  "success": true,
  "time_series": [...],
  "summary": { ... },
  "exp_capacity_params": { ... }
}
```

## Key Features

1. **Self-Contained**: No external dependencies beyond Vercel
2. **Real Python Computation**: Uses actual model code, not JavaScript approximations
3. **Serverless**: Automatically scales with usage
4. **Fast**: Optimized for quick cold starts

## Differences from Local Development

1. **No Flask Server**: Uses Vercel serverless functions instead
2. **No Virtual Environment**: Vercel manages Python dependencies
3. **No Subprocess Calls**: Direct Python function execution
4. **Stateless**: No session management (state handled in frontend)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all Python files are in the Next.js root directory
2. **Path Issues**: Use relative paths from the serverless function perspective
3. **Memory Limits**: Vercel has memory limits for serverless functions
4. **Timeout Issues**: Complex computations may hit Vercel's execution time limits

### Debugging

1. **Local Testing**: Run Python functions directly:
   ```bash
   python api/compute.py
   python api/parameter-config.py
   ```

2. **Vercel Logs**: Use `vercel logs` to see runtime errors

3. **Development Mode**: Use `vercel dev` for local development with serverless functions

## Performance Considerations

- **Cold Start**: First request may be slower due to Python initialization
- **Memory Usage**: NumPy/SciPy arrays can use significant memory
- **Computation Time**: Complex model runs should complete within Vercel's timeout limits

## Future Enhancements

1. **Caching**: Add response caching for common parameter sets
2. **Error Handling**: More robust error handling and user feedback
3. **Optimization**: Pre-compute common scenarios to reduce latency
4. **Monitoring**: Add performance monitoring and logging
