## CS4999-Capstone-Project

### Vector Search with Postgres & pgvector

This project is an example of how to build a **semantic search** system.
It is a good starting-point for the team to build upon our research on Vector Databases and AI-powered searching.

---

### What does it do?
- Imagine you have a huge library of documents. A normal search (like a 'Ctrl+F') can only find the exact words you type.
- This project builds something much smarter. It's like having a librarian who understands the *meaning* behind your questions. You can ask "What is the biggest city in Japan?" and the librarian can find a passage that says "Tokyo is Japan's most populous metropolis", even though you didn't use the words "populous" or "metropolis".
- This is done by converting text into special lists of numbers called **vector embeddings**. Text with similar meanings will have similar numbers. Our database is special because it's good at finding numbers that are close to each other, which is how we find similar text!

---

### How it works?
The project is split into two main parts:

1.  **`seed.py` (The Organizer)**
    * This script acts like an organizer for our library.
    * It reads thousands of passages from the mini-Wikipedia dataset.
    * It uses an AI model to convert each passage into a vector embedding (its unique "meaning fingerprint").
    * Finally, it carefully stores each passage and its fingerprint in our PostgreSQL database.

2.  **`sim_search.py` (The Search Tool)**
    * This script is the search bar for our smart library.
    * It asks you for a query (a question or a topic).
    * It converts your query into its own vector fingerprint.
    * It then asks the database: "Find me the stored passages that have the most similar fingerprints to this one."
    * It displays the top matching passages it finds.

### Prerequisites

Before you begin, you'll need a couple of things installed on your system:

* **Python 3.8** or higher.
* **PostgreSQL** (version 11 or higher).
* The **`pgvector`** extension for PostgreSQL. Make sure it's enabled in the database you plan to use. You can find installation instructions [here](https://github.com/pgvector/pgvector).

---

### Guide:

- Step 1: Clone the Repository

First, get the code onto your local machine.
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

- Step 2: Set up a Venv & Install Dependencies
It's good practice to use a virtual environment to keep project dependencies separate.
```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install all the required Python packages
pip install -r requirements.txt
```

- Step 3: Configure your database connection
The scripts need to know how to connect to your database.

1. Create a file named .env in the project root.
2. Copy the contents of the .env.example file into it.
3. Open your new .env file and replace the placeholder values with your actual PostgreSQL credentials.

```# .env file
DB_NAME="your_db_name"
DB_USER="your_db_user"
DB_PASSWORD="your_super_secret_password"
DB_HOST="localhost"
DB_PORT="5432"
```

- Step 4: Load and Embed the Data
Now it's time to run the "organizer" script. This will download the dataset, process all the text, and save it to your database.

Note: The first time you run this, it will download the AI model (a few hundred MB) and then process all the data. This might take several minutes depending on your internet and computer speed.

```bash
python seed.py
```
You should see output confirming the connection, table setup and insertion progress.

- Step 5: Search your Data
Once the seeding process is complete, you can start searching. Run the search script and type in a query when prompted.

```bash
python sim_search.py
```

It will ask for your input. Try a few different questions to see what it finds!

```bash
Enter your query to find relevant passages: What is the capital of France?

--- Search Results ---
Query: What is the capital of France?

1. Passage ID: 123
   Content: Paris is the capital and most populous city of France...
   Similarity Score: 0.9123
...
```