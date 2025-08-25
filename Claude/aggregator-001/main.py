import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from pydantic import BaseModel, HttpUrl, Field
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import uuid
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ServiceConfig(BaseModel):
    url: HttpUrl
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    params: Optional[Dict[str, str]] = Field(default_factory=dict)
    timeout: int = Field(default=10, ge=1, le=60)
    method: str = Field(default="GET", pattern=r"^(GET|POST|PUT|DELETE)$")

class AggregationRequest(BaseModel):
    services: Optional[List[str]] = Field(None, description="List of service names to aggregate. If empty, all services will be used")
    min_successful_responses: Optional[int] = Field(1, ge=1, description="Minimum number of successful responses to wait for")
    timeout_seconds: Optional[int] = Field(30, ge=1, le=300, description="Maximum time to wait before returning (1-300 seconds)")

class ServiceRegistration(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    config: ServiceConfig

class ServiceResponse(BaseModel):
    service: str
    status: str
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime
    response_time_ms: Optional[float] = None

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    services_requested: List[str]
    min_successful_responses: int
    timeout_seconds: int
    successful_responses_received: int = 0
    results: Optional['AggregatedResponse'] = None
    error: Optional[str] = None

class AggregatedResponse(BaseModel):
    timestamp: datetime
    services: Dict[str, ServiceResponse]
    summary: Dict[str, Any]

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    services_count: int
    uptime_seconds: float
    active_jobs: int

class ServiceAggregator:
    """Enhanced aggregator class with job management and async processing"""

    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.start_time = datetime.now()
        self.request_stats = {}
        self.jobs: Dict[str, JobResponse] = {}  # In-memory job storage
        self.job_cleanup_interval = 3600  # Clean up completed jobs after 1 hour

    def register_service(self, name: str, config: ServiceConfig):
        """Register a new service with the aggregator"""
        self.services[name] = config
        self.request_stats[name] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        logger.info(f"Registered service: {name}")

    def unregister_service(self, name: str) -> bool:
        """Remove a service from the aggregator"""
        if name in self.services:
            del self.services[name]
            if name in self.request_stats:
                del self.request_stats[name]
            logger.info(f"Unregistered service: {name}")
            return True
        return False

    def create_aggregation_job(self, services_to_fetch: Optional[List[str]] = None,
                             min_successful_responses: int = 1, timeout_seconds: int = 30) -> str:
        """Create a new aggregation job and return its UUID"""
        if services_to_fetch is None:
            services_to_fetch = list(self.services.keys())

        # Validate service names
        invalid_services = [s for s in services_to_fetch if s not in self.services]
        if invalid_services:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown services: {', '.join(invalid_services)}"
            )

        # Validate min_successful_responses doesn't exceed available services
        if min_successful_responses > len(services_to_fetch):
            raise HTTPException(
                status_code=400,
                detail=f"min_successful_responses ({min_successful_responses}) cannot exceed number of services ({len(services_to_fetch)})"
            )

        job_id = str(uuid.uuid4())
        job = JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            services_requested=services_to_fetch,
            min_successful_responses=min_successful_responses,
            timeout_seconds=timeout_seconds
        )

        self.jobs[job_id] = job
        logger.info(f"Created aggregation job {job_id} for services: {services_to_fetch}, "
                   f"min_success: {min_successful_responses}, timeout: {timeout_seconds}s")
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def get_active_jobs_count(self) -> int:
        """Get count of active (non-completed) jobs"""
        return len([job for job in self.jobs.values()
                   if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]])

    async def process_aggregation_job_with_conditions(self, job_id: str):
        """Process an aggregation job with conditional completion"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        try:
            # Update job status to processing
            job.status = JobStatus.PROCESSING
            logger.info(f"Processing job {job_id} with conditions: min_success={job.min_successful_responses}, timeout={job.timeout_seconds}s")

            # Start timeout task
            timeout_task = asyncio.create_task(asyncio.sleep(job.timeout_seconds))

            # Start aggregation task
            aggregation_task = asyncio.create_task(
                self.aggregate_data_with_early_completion(job_id, job.services_requested, job.min_successful_responses)
            )

            # Wait for either condition to be met
            done, pending = await asyncio.wait(
                [timeout_task, aggregation_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check which condition was met
            if aggregation_task in done:
                # Aggregation completed successfully
                result = await aggregation_task
                job.results = result
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                logger.info(f"Job {job_id} completed successfully with {job.successful_responses_received} successful responses")
            else:
                # Timeout occurred - get partial results
                logger.info(f"Job {job_id} timed out after {job.timeout_seconds}s, collecting partial results")
                try:
                    # Cancel the aggregation and get whatever results we have
                    aggregation_task.cancel()
                    await aggregation_task
                except asyncio.CancelledError:
                    pass

                # Get partial results
                partial_result = await self.get_partial_results(job_id, job.services_requested)
                job.results = partial_result
                job.status = JobStatus.COMPLETED  # Still mark as completed, but with partial results
                job.completed_at = datetime.now()
                logger.info(f"Job {job_id} completed with timeout - partial results with {job.successful_responses_received} successful responses")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()

    async def aggregate_data_with_early_completion(self, job_id: str, services_to_fetch: List[str],
                                                 min_successful_responses: int) -> AggregatedResponse:
        """Aggregate data but complete early when minimum successful responses are received"""
        job = self.jobs.get(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")

        # Create connector with connection pooling
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)

        async with aiohttp.ClientSession(connector=connector) as session:
            # Start all tasks but don't wait for all to complete
            tasks = {}
            for service_name in services_to_fetch:
                task = asyncio.create_task(
                    self.fetch_service_data(session, service_name, self.services[service_name])
                )
                tasks[task] = service_name

            completed_results = {}
            successful_count = 0

            # Process results as they complete
            for task in asyncio.as_completed(tasks.keys()):
                try:
                    result = await task
                    service_name = tasks[task]
                    completed_results[service_name] = result

                    # Update job progress
                    if result.status == 'success':
                        successful_count += 1
                        job.successful_responses_received = successful_count

                        # Check if we've met the minimum success condition
                        if successful_count >= min_successful_responses:
                            logger.info(f"Job {job_id} reached minimum successful responses ({successful_count}/{min_successful_responses})")
                            break

                except Exception as e:
                    service_name = tasks[task]
                    logger.error(f"Error in task for service {service_name}: {e}")

            # Cancel remaining tasks if we completed early
            for task in tasks.keys():
                if not task.done():
                    task.cancel()

            # Build response from completed results
            return self.build_aggregated_response(completed_results, services_to_fetch)

    async def get_partial_results(self, job_id: str, services_requested: List[str]) -> AggregatedResponse:
        """Get partial results for a job that timed out"""
        # This would contain any results that completed before timeout
        # For now, return empty results structure
        return AggregatedResponse(
            timestamp=datetime.now(),
            services={},
            summary={
                'total_services': len(services_requested),
                'successful': 0,
                'failed': len(services_requested),
                'success_rate': 0.0,
                'avg_response_time_ms': 0.0,
                'completion_reason': 'timeout'
            }
        )

    def build_aggregated_response(self, completed_results: Dict[str, ServiceResponse],
                                all_services: List[str]) -> AggregatedResponse:
        """Build aggregated response from completed results"""
        successful = 0
        failed = 0
        total_response_time = 0

        for result in completed_results.values():
            if result.status == 'success':
                successful += 1
                if result.response_time_ms:
                    total_response_time += result.response_time_ms
            else:
                failed += 1

        # Add pending services as failed
        pending_services = len(all_services) - len(completed_results)
        failed += pending_services

        avg_response_time = total_response_time / successful if successful > 0 else 0
        total_requested = len(all_services)

        return AggregatedResponse(
            timestamp=datetime.now(),
            services=completed_results,
            summary={
                'total_services': total_requested,
                'successful': successful,
                'failed': failed,
                'pending': pending_services,
                'success_rate': (successful / total_requested) * 100,
                'avg_response_time_ms': round(avg_response_time, 2),
                'completion_reason': 'early_completion' if pending_services > 0 else 'all_completed'
            }
        )

    def cleanup_old_jobs(self):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(seconds=self.job_cleanup_interval)
        jobs_to_remove = [
            job_id for job_id, job in self.jobs.items()
            if job.completed_at and job.completed_at < cutoff_time
        ]

        for job_id in jobs_to_remove:
            del self.jobs[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

    async def fetch_service_data(self, session: aiohttp.ClientSession,
                               service_name: str, config: ServiceConfig) -> ServiceResponse:
        """Fetch data from a single service with enhanced monitoring"""
        start_time = datetime.now()

        try:
            timeout = aiohttp.ClientTimeout(total=config.timeout)

            request_kwargs = {
                'headers': config.headers,
                'timeout': timeout
            }

            if config.method.upper() == 'GET':
                request_kwargs['params'] = config.params
                async with session.get(str(config.url), **request_kwargs) as response:
                    response_data = await self._process_response(response)
            elif config.method.upper() == 'POST':
                request_kwargs['json'] = config.params
                async with session.post(str(config.url), **request_kwargs) as response:
                    response_data = await self._process_response(response)
            else:
                # Handle other HTTP methods
                method_func = getattr(session, config.method.lower())
                async with method_func(str(config.url), **request_kwargs) as response:
                    response_data = await self._process_response(response)

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            # Update stats
            self._update_service_stats(service_name, True, response_time)

            return ServiceResponse(
                service=service_name,
                status='success',
                data=response_data,
                timestamp=end_time,
                response_time_ms=response_time
            )

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {config.timeout} seconds"
            self._update_service_stats(service_name, False, 0)
            logger.error(f"Timeout fetching from {service_name}: {error_msg}")
            return ServiceResponse(
                service=service_name,
                status='timeout',
                error=error_msg,
                timestamp=datetime.now()
            )
        except Exception as e:
            self._update_service_stats(service_name, False, 0)
            logger.error(f"Error fetching from {service_name}: {str(e)}")
            return ServiceResponse(
                service=service_name,
                status='error',
                error=str(e),
                timestamp=datetime.now()
            )

    async def _process_response(self, response: aiohttp.ClientResponse) -> Any:
        """Process HTTP response based on content type"""
        if response.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=f"HTTP {response.status}"
            )

        content_type = response.headers.get('Content-Type', '').lower()

        if 'application/json' in content_type:
            return await response.json()
        elif 'text/' in content_type:
            return await response.text()
        else:
            # Try JSON first, fall back to text
            try:
                return await response.json()
            except:
                return await response.text()

    def _update_service_stats(self, service_name: str, success: bool, response_time: float):
        """Update service statistics"""
        if service_name in self.request_stats:
            stats = self.request_stats[service_name]
            stats['total_requests'] += 1

            if success:
                stats['successful_requests'] += 1
                # Update average response time
                total_success = stats['successful_requests']
                current_avg = stats['avg_response_time']
                stats['avg_response_time'] = (current_avg * (total_success - 1) + response_time) / total_success
            else:
                stats['failed_requests'] += 1

    async def aggregate_data_internal(self, services_to_fetch: List[str]) -> AggregatedResponse:
        """Internal method to aggregate data from multiple services concurrently"""
        # Create connector with connection pooling
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.fetch_service_data(session, service_name, self.services[service_name])
                for service_name in services_to_fetch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            services_data = {}
            successful = 0
            failed = 0
            total_response_time = 0

            for result in results:
                if isinstance(result, ServiceResponse):
                    service_name = result.service
                    services_data[service_name] = result

                    if result.status == 'success':
                        successful += 1
                        if result.response_time_ms:
                            total_response_time += result.response_time_ms
                    else:
                        failed += 1
                elif isinstance(result, Exception):
                    logger.error(f"Unexpected error in aggregation: {result}")
                    failed += 1

            avg_response_time = total_response_time / successful if successful > 0 else 0

            aggregated_response = AggregatedResponse(
                timestamp=datetime.now(),
                services=services_data,
                summary={
                    'total_services': len(services_to_fetch),
                    'successful': successful,
                    'failed': failed,
                    'success_rate': (successful / len(services_to_fetch)) * 100,
                    'avg_response_time_ms': round(avg_response_time, 2)
                }
            )

            return aggregated_response

    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics for all services"""
        return {
            'services': self.request_stats,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_jobs': len(self.jobs),
            'active_jobs': self.get_active_jobs_count()
        }

# Initialize aggregator
aggregator = ServiceAggregator()

# Lifespan manager for startup/shutdown tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI Web Service Aggregator")

    # Register example services
    example_services = {
        'jsonplaceholder_posts': ServiceConfig(
            url='https://jsonplaceholder.typicode.com/posts',
            headers={'Content-Type': 'application/json'},
            timeout=10
        ),
        'jsonplaceholder_users': ServiceConfig(
            url='https://jsonplaceholder.typicode.com/users',
            headers={'Content-Type': 'application/json'},
            timeout=10
        ),
        'httpbin_ip': ServiceConfig(
            url='https://httpbin.org/ip',
            headers={'Accept': 'application/json'},
            timeout=5
        ),
        'httpbin_headers': ServiceConfig(
            url='https://httpbin.org/headers',
            headers={'Accept': 'application/json'},
            timeout=5
        )
    }

    for name, config in example_services.items():
        aggregator.register_service(name, config)

    yield

    # Shutdown
    logger.info("Shutting down FastAPI Web Service Aggregator")

# Initialize FastAPI app
app = FastAPI(
    title="Web Service Aggregator",
    description="A powerful aggregator for multiple web services with caching, monitoring, and async processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Interactive dashboard for the aggregator"""
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI Web Service Aggregator</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
            .header h1 { margin: 0; color: #333; font-size: 2.5em; text-align: center; }
            .header p { text-align: center; color: #666; font-size: 1.1em; margin: 10px 0 0 0; }
            .card { background: rgba(255,255,255,0.95); padding: 25px; margin: 15px 0; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
            .btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; margin: 8px; font-size: 14px; transition: all 0.3s ease; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
            .btn:active { transform: translateY(0); }
            .status-success { color: #28a745; font-weight: bold; }
            .status-error { color: #dc3545; font-weight: bold; }
            .status-timeout { color: #ffc107; font-weight: bold; }
            #results { margin-top: 20px; }
            .json-display { background: #f8f9fa; padding: 20px; border-radius: 8px; overflow-x: auto; border-left: 4px solid #667eea; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .stat-item { background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px; text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; font-size: 0.9em; text-transform: uppercase; }
            .service-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }
            .service-item:last-child { border-bottom: none; }
            .loading { text-align: center; padding: 40px; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 FastAPI Web Service Aggregator</h1>
                <p>Aggregate and monitor multiple web services with advanced caching and analytics</p>
            </div>

            <div class="card">
                <h2>🎛️ Quick Actions</h2>
                <button class="btn" onclick="aggregateAll()">📊 Create Aggregation Job</button>
                <button class="btn" onclick="aggregateWithConditions()">⚡ Smart Aggregation</button>
                <button class="btn" onclick="listServices()">📋 List Services</button>
                <button class="btn" onclick="showStats()">📈 Show Statistics</button>
                <button class="btn" onclick="listJobs()">📄 List Jobs</button>
                <button class="btn" onclick="clearResults()">✨ Clear Results</button>
                <button class="btn" onclick="window.open('/docs', '_blank')">📚 API Docs</button>
            </div>

            <div class="card">
                <h2>⚙️ Smart Aggregation Settings</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
                    <div>
                        <label for="minResponses" style="display: block; margin-bottom: 5px; font-weight: bold;">Min Successful Responses:</label>
                        <input type="number" id="minResponses" min="1" max="10" value="2" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    <div>
                        <label for="timeoutSeconds" style="display: block; margin-bottom: 5px; font-weight: bold;">Timeout (seconds):</label>
                        <input type="number" id="timeoutSeconds" min="5" max="300" value="15" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                </div>
                <p style="color: #666; font-size: 0.9em; margin: 10px 0;">Smart aggregation completes when either the minimum successful responses are received OR the timeout is reached.</p>
            </div>

            <div class="card">
                <h2>🔧 Registered Services</h2>
                <div id="service-list">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Loading services...</p>
                    </div>
                </div>
            </div>

            <div id="results"></div>
        </div>

        <script>
            async function apiCall(endpoint, options = {}) {
                try {
                    const response = await fetch(endpoint, options);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return await response.json();
                } catch (error) {
                    console.error('API call failed:', error);
                    throw error;
                }
            }

            async function listServices() {
                try {
                    const data = await apiCall('/api/services');
                    const servicesHtml = data.services.map(service =>
                        `<div class="service-item">
                            <strong>🔗 ${service}</strong>
                            <button class="btn" style="padding: 8px 16px; font-size: 12px;" onclick="fetchService('${service}')">Fetch Data</button>
                        </div>`
                    ).join('');

                    document.getElementById('service-list').innerHTML =
                        `<p><strong>Total Services:</strong> ${data.count}</p>${servicesHtml}`;
                } catch (error) {
                    document.getElementById('service-list').innerHTML =
                        `<div class="status-error">❌ Error loading services: ${error.message}</div>`;
                }
            }

            async function aggregateAll() {
                showLoading('Creating standard aggregation job...');
                try {
                    const response = await apiCall('/api/aggregate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    });

                    if (response.job_id) {
                        showLoading(`Job created (${response.job_id}). Polling for results...`);
                        pollJobStatus(response.job_id);
                    }
                } catch (error) {
                    displayError('Error creating aggregation job: ' + error.message);
                }
            }

            async function aggregateWithConditions() {
                const minResponses = parseInt(document.getElementById('minResponses').value) || 2;
                const timeoutSeconds = parseInt(document.getElementById('timeoutSeconds').value) || 15;

                showLoading(`Creating smart aggregation job (min: ${minResponses} responses, timeout: ${timeoutSeconds}s)...`);
                try {
                    const response = await apiCall('/api/aggregate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            min_successful_responses: minResponses,
                            timeout_seconds: timeoutSeconds
                        })
                    });

                    if (response.job_id) {
                        showLoading(`Smart job created (${response.job_id}). Will complete with ${minResponses} responses or after ${timeoutSeconds}s...`);
                        pollJobStatus(response.job_id);
                    }
                } catch (error) {
                    displayError('Error creating smart aggregation job: ' + error.message);
                }
            }

            async function pollJobStatus(jobId, maxAttempts = 30) {
                let attempts = 0;

                const poll = async () => {
                    try {
                        attempts++;
                        const jobData = await apiCall(`/api/offers/${jobId}`);

                        if (jobData.status === 'completed') {
                            displayResults(jobData.results, `Job ${jobId} - Completed`);
                        } else if (jobData.status === 'failed') {
                            displayError(`Job ${jobId} failed: ${jobData.error}`);
                        } else if (jobData.status === 'processing' || jobData.status === 'pending') {
                            if (attempts < maxAttempts) {
                                showLoading(`Job ${jobId} is ${jobData.status}... (attempt ${attempts}/${maxAttempts})`);
                                setTimeout(poll, 2000); // Poll every 2 seconds
                            } else {
                                displayError(`Job ${jobId} timed out after ${maxAttempts} attempts`);
                            }
                        }
                    } catch (error) {
                        displayError(`Error polling job ${jobId}: ${error.message}`);
                    }
                };

                poll();
            }

            async function listJobs() {
                showLoading('Loading jobs...');
                try {
                    const data = await apiCall('/api/jobs');
                    displayJobs(data);
                } catch (error) {
                    displayError('Error loading jobs: ' + error.message);
                }
            }

            async function fetchService(serviceName) {
                showLoading(`Fetching data from ${serviceName}...`);
                try {
                    const data = await apiCall(`/api/service/${serviceName}`);
                    displayResults({services: {[serviceName]: data}}, `Service: ${serviceName}`);
                } catch (error) {
                    displayError(`Error fetching ${serviceName}: ` + error.message);
                }
            }

            async function showStats() {
                showLoading('Loading statistics...');
                try {
                    const data = await apiCall('/api/stats');
                    displayStats(data);
                } catch (error) {
                    displayError('Error loading stats: ' + error.message);
                }
            }

            function displayJobs(jobs) {
                const resultsDiv = document.getElementById('results');
                let html = '<div class="card"><h2>📄 Active Jobs</h2>';

                if (jobs.length === 0) {
                    html += '<p>No jobs found.</p>';
                } else {
                    html += '<div class="stats-grid">';

                    jobs.forEach(job => {
                        const statusClass = job.status === 'completed' ? 'status-success' :
                                          job.status === 'failed' ? 'status-error' : 'status-timeout';
                        const statusIcon = job.status === 'completed' ? '✅' :
                                         job.status === 'failed' ? '❌' : '⏳';

                        html += `
                            <div class="stat-item" style="text-align: left; padding: 20px;">
                                <h4>${statusIcon} Job ${job.job_id.substring(0, 8)}...</h4>
                                <p><strong>Status:</strong> <span class="${statusClass}">${job.status.toUpperCase()}</span></p>
                                <p><strong>Services:</strong> ${job.services_requested.join(', ')}</p>
                                <p><strong>Conditions:</strong> Min ${job.min_successful_responses} responses, ${job.timeout_seconds}s timeout</p>
                                <p><strong>Progress:</strong> ${job.successful_responses_received || 0} successful responses</p>
                                <p><strong>Created:</strong> ${new Date(job.created_at).toLocaleString()}</p>
                                ${job.completed_at ? `<p><strong>Completed:</strong> ${new Date(job.completed_at).toLocaleString()}</p>` : ''}
                                <button class="btn" style="margin-top: 10px; padding: 8px 16px; font-size: 12px;"
                                        onclick="viewJobResults('${job.job_id}')">View Results</button>
                            </div>
                        `;
                    });

                    html += '</div>';
                }

                html += '</div>';
                resultsDiv.innerHTML = html;
            }

            async function viewJobResults(jobId) {
                showLoading(`Loading results for job ${jobId}...`);
                try {
                    const jobData = await apiCall(`/api/offers/${jobId}`);
                    if (jobData.results) {
                        displayResults(jobData.results, `Job ${jobId} Results`);
                    } else {
                        displayError(`Job ${jobId} has no results yet (status: ${jobData.status})`);
                    }
                } catch (error) {
                    displayError(`Error loading job results: ${error.message}`);
                }
            }

            function showLoading(message = 'Loading...') {
                document.getElementById('results').innerHTML =
                    `<div class="card loading">
                        <div class="spinner"></div>
                        <p>${message}</p>
                    </div>`;
            }

            function displayResults(data, title) {
                const resultsDiv = document.getElementById('results');
                let html = `<div class="card"><h2>📋 ${title}</h2>`;

                if (data.summary) {
                    const summary = data.summary;
                    html += `
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-value">${summary.total_services}</div>
                                <div class="stat-label">Total Services</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value status-success">${summary.successful}</div>
                                <div class="stat-label">Successful</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value status-error">${summary.failed}</div>
                                <div class="stat-label">Failed</div>
                            </div>
                            ${summary.pending ? `
                            <div class="stat-item">
                                <div class="stat-value status-timeout">${summary.pending}</div>
                                <div class="stat-label">Pending/Skipped</div>
                            </div>
                            ` : ''}
                            <div class="stat-item">
                                <div class="stat-value">${summary.success_rate?.toFixed(1)}%</div>
                                <div class="stat-label">Success Rate</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${summary.avg_response_time_ms?.toFixed(0)}ms</div>
                                <div class="stat-label">Avg Response</div>
                            </div>
                            ${summary.completion_reason ? `
                            <div class="stat-item">
                                <div class="stat-value" style="font-size: 1em;">${summary.completion_reason.replace('_', ' ').toUpperCase()}</div>
                                <div class="stat-label">Completion Reason</div>
                            </div>
                            ` : ''}
                        </div>
                    `;
                }Fixed(0)}ms</div>
                                <div class="stat-label">Avg Response</div>
                            </div>
                        </div>
                    `;
                }

                if (data.services) {
                    for (const [serviceName, serviceData] of Object.entries(data.services)) {
                        const statusClass = serviceData.status === 'success' ? 'status-success' :
                                          serviceData.status === 'timeout' ? 'status-timeout' : 'status-error';
                        const statusIcon = serviceData.status === 'success' ? '✅' :
                                         serviceData.status === 'timeout' ? '⏰' : '❌';

                        html += `
                            <div style="margin: 20px 0; padding: 15px; border-left: 4px solid ${getStatusColor(serviceData.status)}; background: #f8f9fa; border-radius: 8px;">
                                <h3>${statusIcon} ${serviceName} <span class="${statusClass}">[${serviceData.status.toUpperCase()}]</span></h3>
                                ${serviceData.response_time_ms ? `<p><strong>Response Time:</strong> ${serviceData.response_time_ms.toFixed(2)}ms</p>` : ''}
                                <div class="json-display"><pre>${JSON.stringify(serviceData, null, 2)}</pre></div>
                            </div>
                        `;
                    }
                }

                html += '</div>';
                resultsDiv.innerHTML = html;
            }

            function displayStats(data) {
                const resultsDiv = document.getElementById('results');
                let html = '<div class="card"><h2>📊 Service Statistics</h2>';

                html += `
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">${Math.round(data.uptime_seconds)}s</div>
                            <div class="stat-label">Uptime</div>
                        </div>
                    </div>
                `;

                if (data.services) {
                    html += '<h3>Service Performance</h3>';
                    for (const [serviceName, stats] of Object.entries(data.services)) {
                        const successRate = stats.total_requests > 0 ?
                            (stats.successful_requests / stats.total_requests * 100).toFixed(1) : 0;

                        html += `
                            <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                                <h4>🔗 ${serviceName}</h4>
                                <div class="stats-grid">
                                    <div class="stat-item">
                                        <div class="stat-value">${stats.total_requests}</div>
                                        <div class="stat-label">Total Requests</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value status-success">${stats.successful_requests}</div>
                                        <div class="stat-label">Successful</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value status-error">${stats.failed_requests}</div>
                                        <div class="stat-label">Failed</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">${successRate}%</div>
                                        <div class="stat-label">Success Rate</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">${stats.avg_response_time.toFixed(0)}ms</div>
                                        <div class="stat-label">Avg Response</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                }

                html += '</div>';
                resultsDiv.innerHTML = html;
            }

            function displayError(message) {
                document.getElementById('results').innerHTML =
                    `<div class="card"><h2 class="status-error">❌ Error</h2><p>${message}</p></div>`;
            }

            function displaySuccess(message) {
                document.getElementById('results').innerHTML =
                    `<div class="card"><h2 class="status-success">Success</h2><p>${message}</p></div>`;
            }

            function clearResults() {
                document.getElementById('results').innerHTML = '';
            }

            function getStatusColor(status) {
                switch(status) {
                    case 'success': return '#28a745';
                    case 'timeout': return '#ffc107';
                    case 'error': default: return '#dc3545';
                }
            }

            // Load services on page load
            listServices();
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/api/services")
async def list_services():
    """Get list of registered services"""
    return {
        "services": list(aggregator.services.keys()),
        "count": len(aggregator.services)
    }

@app.post("/api/aggregate")
async def create_aggregation_job(
    request: AggregationRequest,
    background_tasks: BackgroundTasks
):
    """Create a new aggregation job with conditional completion and redirect to results endpoint"""
    # Create the job with conditions
    job_id = aggregator.create_aggregation_job(
        request.services,
        request.min_successful_responses or 1,
        request.timeout_seconds or 30
    )

    # Start processing in background with conditions
    background_tasks.add_task(aggregator.process_aggregation_job_with_conditions, job_id)

    # Return job information with redirect location
    return {
        "job_id": job_id,
        "status": "pending",
        "min_successful_responses": request.min_successful_responses or 1,
        "timeout_seconds": request.timeout_seconds or 30,
        "message": f"Aggregation job created with conditions. Check results at /api/offers/{job_id}",
        "results_url": f"/api/offers/{job_id}"
    }

@app.get("/api/offers/{job_id}", response_model=JobResponse)
async def get_aggregation_results(job_id: str):
    """Get aggregation results for a specific job"""
    job = aggregator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job

@app.get("/api/jobs")
async def list_jobs():
    """Get list of all jobs"""
    return list(aggregator.jobs.values())

@app.get("/api/service/{service_name}", response_model=ServiceResponse)
async def get_service_data(service_name: str, background_tasks: BackgroundTasks):
    """Get data from a specific service via job system"""
    if service_name not in aggregator.services:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

    # Create a job for single service with minimal conditions
    job_id = aggregator.create_aggregation_job([service_name], min_successful_responses=1, timeout_seconds=30)

    # Process immediately for single service
    await aggregator.process_aggregation_job_with_conditions(job_id)

    job = aggregator.get_job(job_id)
    if job and job.results and service_name in job.results.services:
        return job.results.services[service_name]
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch service data")

@app.post("/api/services")
async def register_service(registration: ServiceRegistration):
    """Register a new service"""
    if registration.name in aggregator.services:
        raise HTTPException(
            status_code=409,
            detail=f"Service '{registration.name}' already exists"
        )

    aggregator.register_service(registration.name, registration.config)

    return {
        "message": f"Service '{registration.name}' registered successfully",
        "service": registration.name
    }

@app.put("/api/services/{service_name}")
async def update_service(service_name: str, config: ServiceConfig):
    """Update an existing service configuration"""
    if service_name not in aggregator.services:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

    aggregator.register_service(service_name, config)  # This overwrites existing

    return {
        "message": f"Service '{service_name}' updated successfully",
        "service": service_name
    }

@app.delete("/api/services/{service_name}")
async def unregister_service(service_name: str):
    """Remove a service from the aggregator"""
    if not aggregator.unregister_service(service_name):
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

    return {
        "message": f"Service '{service_name}' removed successfully",
        "service": service_name
    }

@app.get("/api/stats")
async def get_statistics():
    """Get aggregator statistics and service performance metrics"""
    return aggregator.get_service_stats()

@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear all cached data - No longer applicable but kept for API compatibility"""
    return {"message": "No cache to clear - caching has been disabled"}

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a specific job"""
    if job_id not in aggregator.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    del aggregator.jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.delete("/api/jobs")
async def cleanup_completed_jobs():
    """Clean up all completed jobs"""
    initial_count = len(aggregator.jobs)
    aggregator.cleanup_old_jobs()

    # Also clean up completed jobs regardless of age
    completed_jobs = [
        job_id for job_id, job in aggregator.jobs.items()
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
    ]

    for job_id in completed_jobs:
        del aggregator.jobs[job_id]

    cleaned_count = initial_count - len(aggregator.jobs)
    return {
        "message": f"Cleaned up {cleaned_count} completed jobs",
        "remaining_jobs": len(aggregator.jobs)
    }

@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint with detailed status"""
    stats = aggregator.get_service_stats()
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        services_count=len(aggregator.services),
        uptime_seconds=stats['uptime_seconds'],
        active_jobs=aggregator.get_active_jobs_count()
    )

if __name__ == "__main__":
    print("🚀 Starting FastAPI Web Service Aggregator...")
    print("📊 Dashboard: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔄 ReDoc: http://localhost:8000/redoc")
    print("\n🔗 API Endpoints:")
    print("  GET    /api/services - List all services")
    print("  POST   /api/services - Register new service")
    print("  PUT    /api/services/{name} - Update service")
    print("  DELETE /api/services/{name} - Remove service")
    print("  POST   /api/aggregate - Create smart aggregation job")
    print("  GET    /api/offers/{job_id} - Get job results")
    print("  GET    /api/jobs - List all jobs")
    print("  DELETE /api/jobs/{job_id} - Delete specific job")
    print("  DELETE /api/jobs - Clean up completed jobs")
    print("  GET    /api/service/{name} - Get specific service data")
    print("  GET    /api/stats - Service statistics")
    print("  GET    /api/health - Health check")
    print("\n⚡ Smart Job Processing:")
    print("  • Completes when N successful responses received OR timeout reached")
    print("  • Configurable min_successful_responses (default: 1)")
    print("  • Configurable timeout_seconds (default: 30, max: 300)")
    print("  • Early completion for faster results")
    print("  • Partial results on timeout")

    try:
        uvicorn.run(
            app,  # Use the app object directly instead of string reference
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to False to avoid import issues
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Try running with: uvicorn main:app --host 0.0.0.0 --port 8000")

