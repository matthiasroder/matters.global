# Railway Deployment Guide for Matters.Global

This guide walks you through deploying Matters.Global to Railway with separate services for the backend and frontend.

## Prerequisites

1. Railway account (https://railway.app)
2. GitHub account with this repository
3. OpenAI API key

## Deployment Steps

### 1. Create Railway Project

1. Visit [Railway](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your matters.global repository
5. Select the `dev` branch

### 2. Set Up Neo4j Database

1. In your Railway project dashboard, click "New Service"
2. Choose "Database" → "Add Neo4j"
3. Note the connection details (Railway will provide these as environment variables)

### 3. Deploy Backend Service

1. Click "New Service" → "GitHub Repo"
2. Select your repository and `dev` branch
3. Railway will detect the `Dockerfile` and `railway.toml` in the root
4. Add environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   NEO4J_URI=neo4j://localhost:7687  # Railway will override this
   NEO4J_USERNAME=neo4j  # Railway will override this  
   NEO4J_PASSWORD=password  # Railway will override this
   PORT=8080
   ```
5. Railway will automatically deploy the Python backend

### 4. Deploy Frontend Service

1. Click "New Service" → "GitHub Repo"
2. Select your repository and `dev` branch
3. Set the "Root Directory" to `ui`
4. Railway will detect the `package.json` and build the React app
5. Update `ui/src/config.ts` to point to your deployed backend:
   ```typescript
   export const WEBSOCKET_ENDPOINT = "wss://your-backend-service.railway.app";
   ```

### 5. Configure Custom Domains (Optional)

1. In each service, go to Settings → Domains
2. Add custom domains or use Railway-provided domains
3. Update CORS settings in your backend if needed

## Environment Variables

### Backend Service
- `OPENAI_API_KEY`: Your OpenAI API key
- `NEO4J_URI`: Provided by Railway Neo4j service
- `NEO4J_USERNAME`: Provided by Railway Neo4j service
- `NEO4J_PASSWORD`: Provided by Railway Neo4j service
- `PORT`: 8080 (or Railway's default)

### Frontend Service
- No additional environment variables needed
- Update `config.ts` to point to backend URL

## Project Structure

```
matters.global/
├── Dockerfile                # Backend container
├── railway.toml              # Backend Railway config
├── requirements.txt          # Python dependencies
├── *.py                      # Backend Python files
├── config/                   # Backend config files
└── ui/
    ├── railway.toml          # Frontend Railway config
    ├── package.json          # Node.js dependencies
    └── src/                  # React source code
```

## Troubleshooting

### Backend Issues
- Check Railway logs for Python import errors
- Verify all environment variables are set
- Ensure Neo4j service is running and accessible

### Frontend Issues
- Verify WebSocket endpoint URL in `config.ts`
- Check CORS settings on backend
- Ensure frontend builds successfully

### Database Issues
- Verify Neo4j service is running
- Check connection strings match Railway-provided values
- Test database connectivity from backend service

## Monitoring

1. Railway provides logs for each service
2. Set up health checks if needed
3. Monitor resource usage in Railway dashboard

## Cost Estimation

- **Hobby Plan**: ~$5-10/month for small usage
- **Pro Plan**: ~$20+/month for production usage
- Costs scale with resource usage (CPU, memory, network)

## Next Steps

1. Test the deployment thoroughly
2. Set up monitoring and alerts
3. Configure backup strategies for Neo4j
4. Consider adding staging environment
5. Set up CI/CD pipeline for automatic deployments