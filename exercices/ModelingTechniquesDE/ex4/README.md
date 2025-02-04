# SQL Query Optimization

## Exercise 1: Index Optimization

You have a sales table with the following schema:

```{sql}
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    customer_id INT,
    product_id INT,
    sale_date DATE,
    amount DECIMAL(10,2)
);
```

Question: Analyze the following query and optimize it using indexes:

```{sql}
SELECT SUM(amount) 
FROM sales 
WHERE sale_date BETWEEN '2024-01-01' AND '2024-06-30';
```

- What index should be created for this query?
- How would the index improve performance?
- How can you check if the index is being used?

## Exercise 2: Query Rewriting Performance

Given a `customers` and `orders` table:

```
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name TEXT
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    order_date DATE,
    total_price DECIMAL(10,2)
);
```

Question:
The following query runs slowly on large datasets:

```
SELECT c.name, SUM(o.total_price)
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.name;
```

- How can you rewrite this query for better performance?
- Would indexing help? If yes, which indexes?

## Exercise 3: Optimizing JOINs with Paritioning

Consider a large orders table:

```
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_date DATE,
    customer_id INT,
    total_price DECIMAL(10,2)
)
PARTITION BY RANGE (order_date);
```

Partitions:

```
CREATE TABLE orders_2024_01 PARTITION OF orders 
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_02 PARTITION OF orders 
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
```

Question: Optimize this slow query

```
SELECT SUM(total_price) 
FROM orders 
WHERE order_date BETWEEN '2024-01-01' AND '2024-03-01';
```
- What improvements do partition pruning bring?
- How do you verify that partitioning is being used?
- Would INDEX(order_date) still be useful?

## Exercise 4. Subquery vs JOIN Optimization

Consider a product sales analysis:

```

SELECT name, 
       (SELECT SUM(amount) 
        FROM sales 
        WHERE sales.product_id = products.id) AS total_sales
FROM products;

```

Question:
- Why is this query inefficient?
- Rewrite it using JOINs for better performance.
- What kind of indexing would improve performance?

## Exercise 5. Window Function Optimization

You need to find the top 3 selling products per month from this dataset:

```

CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    product_id INT,
    sale_date DATE,
    amount DECIMAL(10,2)
);

```

Question: Optimize this query

```

SELECT product_id, sale_date, 
       SUM(amount) OVER (PARTITION BY EXTRACT(MONTH FROM sale_date) ORDER BY SUM(amount) DESC) as rank
FROM sales
LIMIT 3;

```

## Exercise 6: Optimize COUNT queries on large datasets

Given a large users table:

```

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

Question: You need to efficiently get the total number of users

```
SELECT COUNT(*) FROM users;
```

- Why is this query slow on large datasets?
- How can you optimize it using materialized views or approximate counts?
- Can an index on id speed up `COUNT(*)`?


## Exercise 7. Optimize DISTINCT Queries

Given the transactions table:

```

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    user_id INT,
    transaction_date DATE,
    amount DECIMAL(10,2)
);

```

Question: The following query runs slowly

`SELECT DISTINCT user_id FROM transactions;`

- Why is this inefficient on large datasets?
- Would an index on user_id improve performance?
- What alternative query can be used to speed it up?

## Exercise 8. Reduce Sorting in Window Functions

Given a sales table:

CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_price DECIMAL(10,2)
);

Question: The following query is slow due to sorting

SELECT customer_id, order_date, 
       SUM(total_price) OVER (PARTITION BY customer_id ORDER BY order_date) as running_total
FROM sales;

Why does this query sort large datasets?
How can you pre-sort data to optimize window functions?
Can an index help, and if so, which one?

4. Optimize Nested Queries
Given a products table:

sql
Copy
Edit
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT,
    category TEXT,
    price DECIMAL(10,2)
);
Question:
Optimize this correlated subquery:

SELECT name, category, price
FROM products p1
WHERE price = (SELECT MAX(price) FROM products p2 WHERE p1.category = p2.category);
Why is this inefficient?
How can you rewrite it using JOINs?
Would a window function be better?

5. Query Optimization with Partitioning

Given a huge page_views table:

CREATE TABLE page_views (
    id SERIAL PRIMARY KEY,
    user_id INT,
    page_url TEXT,
    view_date DATE
)
PARTITION BY RANGE (view_date);

Question: This query is slow

SELECT COUNT(*) FROM page_views WHERE view_date BETWEEN '2024-01-01' AND '2024-06-30';
How does partition pruning help here?
How do you check if the optimizer is scanning all partitions?
What index strategy should be applied?

6. Optimize Multi-Table JOIN Queries
You have these tables:

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    order_date DATE
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(id),
    product_id INT,
    quantity INT
);

Question: This query is slow

SELECT u.name, COUNT(o.id)
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
GROUP BY u.name;
What indexes should be added?
Can you rewrite the query for better performance?
What role does denormalization play in optimization?

7. Optimizing Large Aggregations
Given a sensor_data table:

CREATE TABLE sensor_data (
    id SERIAL PRIMARY KEY,
    sensor_id INT,
    measurement FLOAT,
    recorded_at TIMESTAMP
);
Question: The following query is slow

SELECT sensor_id, AVG(measurement)
FROM sensor_data
WHERE recorded_at >= NOW() - INTERVAL '7 days'
GROUP BY sensor_id;
How can indexes improve performance?
Would materialized views help?
Could a hybrid approach (pre-aggregated tables + real-time data) be better?

8. Optimize JOINs with Large Tables
You have a big logs table:

CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    user_id INT,
    event TEXT,
    event_date TIMESTAMP
);
Question:
This query is slow:

SELECT u.name, COUNT(l.id)
FROM users u
LEFT JOIN logs l ON u.id = l.user_id
WHERE l.event_date >= NOW() - INTERVAL '30 days'
GROUP BY u.name;
How can you optimize this JOIN?
Would partitioning help?
How can you use bitmap indexes or index-organized tables?

9. Optimize Real-Time Reporting Queries
Given a clicks table:

CREATE TABLE clicks (
    id SERIAL PRIMARY KEY,
    user_id INT,
    page_url TEXT,
    clicked_at TIMESTAMP
);
Question:
The following real-time query is slow:

SELECT COUNT(*)
FROM clicks
WHERE clicked_at >= NOW() - INTERVAL '1 hour';
How can you cache results efficiently?
Would pre-aggregating data help?
How would you use indexing and partitioning together?

10. Optimizing Full-Text Search Queries
You have a products table:

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT
);
Question:
Optimize this slow query:

SELECT * FROM products WHERE description ILIKE '%organic%';

- Why is this slow?
- How can full-text indexing (TSVECTOR, TSQUERY) improve performance?
- Would a materialized view help?



