# Storage Cost Optimization Exercices

## Exercise 1: Choose the Right Storage Class

Scenario: Your company stores daily log files (100 TB per month) in Amazon S3 Standard, but only 1% of logs are accessed within 30 days.

Task: Identify a better storage class to reduce costs while maintaining access to recent logs.
Bonus: Calculate the estimated savings.
Hint
Use S3 Intelligent-Tiering, S3 Glacier, or S3 Lifecycle Policies.
Consider retrieval costs when choosing Glacier.

## Exercise 2: Reduce Data Transfer Costs

Scenario: You have a multi-region architecture where your data pipeline moves 1 TB daily from AWS us-east-1 to eu-west-1. Your data warehouse is in us-east-1.

Task: Propose ways to reduce inter-region data transfer costs.
Bonus: Estimate how much you can save by keeping all services in the same region.
Hint
Avoid unnecessary inter-region data transfers.
Use VPC Endpoints or CloudFront to reduce egress costs.
Compute Cost Optimization Exercises

## Exercise 3: Optimize EC2 Costs
Scenario: Your company runs 20 EC2 instances (m5.large) for a batch job that runs 6 hours per day.

Task: Identify cheaper alternatives without affecting performance.
Bonus: Estimate the cost difference.
Hint
Use Spot Instances for non-critical workloads.
Consider AWS Lambda or Fargate for short-lived tasks.
Use Savings Plans or Reserved Instances for predictable workloads.

## Exercise 4: Serverless Cost Optimization
Scenario: Your AWS Lambda function processes 5 million invocations per day with an average execution time of 2 seconds, using 512MB of memory.

Task: Suggest optimizations to reduce cost while maintaining performance.
Bonus: Compute cost savings after reducing execution time to 1 second.
Hint
Optimize code efficiency to reduce execution time.
Use Provisioned Concurrency if the cold start is a problem.
Check if lower memory allocation is sufficient.

# Database Cost Optimization Exercises

## Exercise 5: Optimize BigQuery Costs

Scenario: Your company runs 10 complex queries daily on BigQuery, scanning 5 TB each.

Task: Propose ways to reduce query costs.
Bonus: Estimate savings by partitioning and clustering the tables.
Hint
Use partitioned tables to scan only relevant data.
Use BI Engine caching for repeated queries.
Check for unused columns in queries to reduce scanned bytes.

## Exercise 6: Right-Size Your Database

Scenario: Your PostgreSQL RDS instance is db.r5.4xlarge, costing $1,000/month, but CPU usage is only 10% on average.

Task: Suggest a cost-effective alternative while maintaining performance.
Bonus: What would happen if you switched to Aurora Serverless?
Hint
Consider downscaling to db.r5.2xlarge or db.r5.large.
Use Aurora Serverless for unpredictable workloads.
Optimize queries and enable auto-scaling.
Networking & Data Transfer Cost Optimization Exercises

## Exercise 7: Reduce Egress Costs in a Data Pipeline

Scenario: Your Apache Spark cluster pulls 100 TB/month from S3 in us-west-2 but runs in us-east-1.

Task: Suggest optimizations to reduce egress costs.
Bonus: What would happen if you used S3 Transfer Acceleration?
Hint
Keep compute and storage in the same region.
Use S3 Select to filter data before transferring.
Consider S3 Transfer Acceleration if moving data across regions.

## Exercise 8: Optimize Cloud CDN for Cost Savings

Scenario: Your company uses CloudFront to serve 1 PB of data/month, but 30% of traffic still goes to origin servers instead of cache.

Task: Optimize caching to reduce origin fetch costs.
Bonus: How can cache TTL tuning help?
Hint
Increase TTL (Time-to-Live) to cache content longer.
Use regional edge caches to avoid unnecessary origin requests.
Enable compression (Gzip, Brotli) to reduce transfer size.
How to Approach These Exercises?
Estimate current costs using AWS Calculator, BigQuery pricing, or Azure Cost Management.
Compare alternative solutions (Spot instances, reserved capacity, partitioning).
Implement cost-saving strategies (Auto-scaling, query optimization, serverless).
Monitor impact using CloudWatch, AWS Cost Explorer, or GCP Cost Management


