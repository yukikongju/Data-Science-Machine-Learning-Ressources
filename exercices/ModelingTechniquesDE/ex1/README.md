# Problem 1 - Identify Fact & Dimension Tables

You are designing a data warehouse for an e-commerce company. You have the following tables:

- Orders (order_id, customer_id, product_id, order_date, quantity, total_price, discount, payment_method)
- Customers (customer_id, name, email, location, date_of_birth, registration_date)
- Products (product_id, name, category, price, supplier_id)
- Suppliers (supplier_id, name, country)
- Order_Reviews (order_id, customer_id, review_score, review_comment)

Questions:

1. Which of these tables are fact tables?
2. Which of these tables are dimension tables?
3. What are the measures (quantitative data) in the fact tables?
4. What are the dimensions (descriptive attributes)?

Answers:
1. The `orders` and `order_reviews` table is a fact table. 
2. The `customers`, `products`, `suppliers` 
3. For orders: qty, price, discount, payment_method; for order_reviews: review_score
4. The dimensions are all the columns in (2)

# Problem 2 - Implement Slowly Chaning Dimensions (SCDs)

Your company tracks product prices over time. The Products table looks like this:

| product_id | name              | category | price | updated_at |
| 1          | yoga mat          | fitness  | 25    | 2024-01-01 |
| 2          | meditation pillow | home     | 30    | 2025-01-03 |

On February 1st, 2024, the price of the Yoga Mat changes from $25 to $30.

Questions:
1. How should we handle this change in the Products table?
   * (SCD Type 1): Overwrite the old value
   * (SCD Type 2): Keep a history of changes
   * (SCD Type 3): Store both old and new values in one row
2. Which method would be best for tracking price changes over time?
3. How would this affect reporting on product prices?

Answers:
1. We probably want to store the table as a type 2 table to make sure we can 
   properly compute the revenues we get. We would probably have a table `orders`
   with columns: order_id, customer_id, date, product_id, qty
    a. In a snowflake table, we would need to join the tables together to compute
    b. In a normalized table, we would have additional columns: price, discount 
       but we still need the price to match 
   The type 2 table would look like this: product_id, name, category, price, start_date, end_date, version
   Whenever an item would be updated, we would need to (1) close the current row 
   and (2) insert the additional row. Ex:

   ```{sql}
   --- get last version
   with last_version as (
       select max(version) 
       from products
       where product_id=1
   )
   
   --- close the existing row
   update products 
   set end_date= CURRENT_DATE
   where product_id=1 and end_date = null and version=last_version
   
   --- insert updated
   insert into product (product_id, name, category, price, start_date, end_date, version) 
   values (1, 'yoga mat', 'fitness', 30, CURRENT_DATE, null, last_version)
   ```

2. Type 2
3. To compute the total price, perform this query

```{sql}
SELECT 
    SUM(p.price * o.qty) as total_revenue
FROM Orders o
JOIN Products p
ON O.product_id = p.product_id
    AND o.date between p.start_date COALESCE(p.end_date, CURRENT_DATE)
```

To get Cost dimension table:

```{sql} 
SELECT
    o.date as date,
    o.order_id as order_id,
    o.qty as qty,
    p.price as item_price,
    o.qty * p.price as total_cost,
FROM Orders o
JOIN Products p
ON o.product_id = p.product_id 
    AND o.date between p.start_date and COALESCE(p.end_date, CURRENT_DATE)
```

To find the top-selling product category

Unoptimized:
```{sql}
with cost_table as (
    SELECT
	o.date as date,
	o.order_id as order_id,
	o.qty as qty,
	p.product_id as product_id,
	p.price as item_price,
	o.qty * p.price as total_cost,
    FROM Orders o
    JOIN Products p
    ON o.product_id = p.product_id 
	AND o.date between p.start_date and COALESCE(p.end_date, CURRENT_DATE)
)

SELECT 
    p.category,
    SUM(c.total_cost)
FROM cost_table c 
JOIN (
    SELECT 
	distinct product_id, name, category
    FROM Products
 ) p 
ON c.product_id = p.product_id
GROUP BY
    p.category
```

Optimized:
```{sql
SELECT  
    RANK() OVER (PARTITION BY category ORDER BY total_revenue DESC) as rank,
    p.product_id,
    p.name,
    p.category,
    SUM(o.qty * p.price) as total_revenue,
FROM Orders o
JOIN Products p
ON o.product_id = p.product_id
    AND o.date between p.start_date and COALESCE(p.end_date, CURRENT_DATE)
GROUP BY 
    p.product_id, p.name, p.category,
ORDER BY
    total_revenue DESC
```

To find the top 3 top-selling product category by month

| year | month | rank | category | sales |

3 steps:
1. compute monthly sales per category
2. rank the categories per month
3. filter top 3

```{sql}

with monthly_sales as (
    SELECT
	EXTRACT(YEAR from o.date) as year,
	EXTRACT(MONTH from o.date) as month,
	SUM(o.qty * p.price) as monthly_sales,
    FROM Orders o 
    JOIN Products p
    ON o.product_id = p.product_id 
	AND o.date between p.start_date and COALESCE(p.end_date, CURRENT_DATE)
), monthly_rank as (
    SELECT
	year, 
	month,
	category,
	total_revenue,
	RANK() OVER (PARTITION BY year, month ORDER BY monthly_sales DESC) as rank
    FROM monthly_sales 
)

SELECT * 
FROM monthly_rank
where rank <= 3
```


# Problem 3 - Create a Fact Table Schema

A fitness app tracks user workouts. You need to design a Workout_Fact table.

Requirements:
- Each workout has a date, user, exercise type, duration, and calories burned.
- Users have attributes like age, gender, and fitness level.
- Exercises have attributes like category (cardio/strength), difficulty, and required equipment.

Tasks:
1. Define the fact table schema.
2. Define the dimension tables needed.
3. What is the grain of the fact table?

Answers:

TODO

# Problem 4 -  Degenerate Dimensions and Junk Dimensions

A food delivery company wants to store order details in a fact table.

Fact Table (Order_Fact):

| order_id | customer_id | restaurant_id | order_date | total_price | payment_type | discount_code | delivery_time |
| 101      | 5001        | 3001          | 2024-02-01 | 30.00       | Credit Card  | SPRING10      | 45 min        |
| 102      | 5002        | 3002          | 2024-02-02 | 25.00       | PayPal       | NULL          | 30 min        |


Questions:
1. Which column in Order_Fact is a degenerate dimension?
2. How could we create a junk dimension from payment_type and discount_code?

Answers:

TODO

# Problem 5 - Star Schema vs Snowflake Schema

A hotel booking system tracks customer reservations.  You have two possible schema designs:

- Star Schema (Denormalized)
    * Fact Table (Reservations_Fact): booking_id, customer_id, hotel_id, checkin_date, checkout_date, total_price
    * Dimension Table (Customers): customer_id, name, email, country
    * Dimension Table (Hotels): hotel_id, name, city, country
- Snowflake Schema (Normalized)
    * Fact Table (Reservations_Fact): booking_id, customer_id, hotel_id, checkin_date, checkout_date, total_price
    * Dimension Table (Customers): customer_id, name, email, country_id
    * Dimension Table (Countries): country_id, country_name
    * Dimension Table (Hotels): hotel_id, name, city_id
    * Dimension Table (Cities): city_id, city_name, country_id

Questions:
1. Which design is easier to query?
2. Which design is better for storage efficiency?
3. Which one would you recommend for a high-performance reporting system?

Answers: 

TODO


