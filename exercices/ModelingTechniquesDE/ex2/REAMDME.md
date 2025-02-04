# Advanced Data Modeling

## Exercise 1 - Designing a Star Schema

Imagine you are tasked with designing a data warehouse schema for an e-commerce platform. The platform tracks customer orders, products, and transactions. You need to design a star schema.

Requirements:
- Fact Table: The fact table will capture the order details, such as quantity ordered, total price, and transaction information.
- Dimension Tables: You need to create the following dimension tables:
    * Product Dimension: Contains details about the products like name, description, and category.
    * Customer Dimension: Contains customer details like name, address, and membership status.
    * Date Dimension: Contains time details about when the order was placed, including day, week, month, and year.
    * Salesperson Dimension: Contains details about the salesperson who handled the order.
Tasks:
1. Create the schema for the fact table and dimension tables.
2. Define primary keys and foreign keys.
3. Decide on the grain of the fact table.
4. Create a SQL DDL script to implement the schema.

## Exercise 2 -  Snowflake Schema

Given the same e-commerce platform scenario, this time, you need to design a snowflake schema. In the snowflake schema, the dimensions are normalized into multiple related tables.

Tasks:
1. Normalize the Product Dimension into multiple related tables (e.g., a product category table, product details table, etc.).
2. Implement the snowflake schema by creating additional dimension tables that normalize the data.
3. Define foreign key relationships between the fact table and the normalized dimension tables.
4. Create a SQL DDL script to implement the snowflake schema.

## Exercise 3: Slowly Changing Dimensions (SCD)

You need to model a customer dimension that includes customer address and membership level. Over time, a customer may change their address or membership level. This is an example of a slowly changing dimension (SCD).

Requirements:
- Implement Type 1 (Overwrite) for the address when a customer updates it.
- Implement Type 2 (Add New Row) for when a customer upgrades their membership level (you need to track changes over time).
- Implement Type 3 (Add New Column) to track changes in both the membership level and address at the same time.

Tasks:
1. Design the schema to handle the above scenarios.
2. Write SQL to insert, update, and manage historical data in the customer dimension using SCD types 1, 2, and 3.

## Exercise 4: Fact Table Granularity

You are working on an inventory management system that tracks product sales across multiple stores. The fact table stores transaction records that capture details of sales.

Requirements:
- The granularity of the fact table is at the daily level.
- You need to capture sales of products at the store level.
- Include the quantity sold and total sales value for each product.

Tasks:
1. Define the grain of the fact table. Should it be at the store level or product level? Should you create a factless fact table to track whether a store has stocked a particular product on a given day?
2. Write a SQL query to calculate the total sales value for each product in the store over the last month.

## Exercise 5: Implementing Bridge Tables

You are working on a sales tracking system. You need to track the products purchased in each order. However, the relationship between products and orders is many-to-many (i.e., each order can have multiple products, and each product can be part of multiple orders).

Requirements:
- Implement a bridge table to resolve this many-to-many relationship.
- The bridge table will contain keys for both products and orders, along with any relevant facts such as quantity and total price.
- The sales fact table will reference the bridge table.

Tasks:
- Design the schema to implement the bridge table.
- Write a SQL DDL script to create the fact table, product table, order table, and bridge table.
- Write a SQL query to find the top-selling products in each region.

## Exercise 6: Data Vault Modeling

You need to implement a Data Vault model for an organization that tracks customer transactions and inventory management. The key concept behind the Data Vault is to focus on creating hubs, links, and satellites.

Requirements:
- Hubs: Create hubs for Customer, Product, and Transaction.
- Links: Create links to capture relationships, such as Customer to Transaction, and Transaction to Product.
- Satellites: Create satellites to store descriptive data for each of the hubs (e.g., customer address, product price history, transaction details).

Tasks:
1. Design the hub and satellite tables for the entities.
2. Create link tables to capture relationships between entities.
3. Write SQL DDL scripts for creating the Data Vault schema.

## Exercise 7: Fact Table with Multiple Time Dimensions

You are designing a data warehouse for a telecom company. The company tracks calls made by customers, including the start time and end time of each call. You need to design a fact table that includes multiple time dimensions.

Requirements:
- You need to track calls at the call start time and call end time.
- Include dimensions like customer, product, and location.
- The fact table needs to store both call duration and call cost.

Tasks:
1. Design the fact table and the associated time dimensions.
2. Create a schema with two time dimensions: start time and end time.
3. Write a SQL query to calculate the average call duration for each customer by month.


