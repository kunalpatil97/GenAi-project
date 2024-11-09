import gradio as gr
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup

# Load the pre-trained model for generating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Placeholder for course data
courses = []


# Step 1: Web Scraping function (Ensure this complies with Analytics Vidhya's policies)
def fetch_courses():
    url = "https://www.analyticsvidhya.com/blog/category/free-courses/"  # Replace with actual URL if different
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    for course in soup.find_all('div', class_="course-item"):  # Adjust selector as per actual HTML structure
        title = course.find('h2').get_text(strip=True)
        description = course.find('p').get_text(strip=True)
        courses.append({
            "title": title,
            "description": description,
            "embedding": model.encode(description, convert_to_tensor=True)
        })


# Fetch courses on script startup
fetch_courses()


# Step 2: Smart Search Function
def smart_search(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = []

    for course in courses:
        similarity = util.pytorch_cos_sim(query_embedding, course["embedding"]).item()
        results.append((course["title"], course["description"], similarity))

    # Sort by similarity score in descending order
    results = sorted(results, key=lambda x: x[2], reverse=True)[:5]

    # Format results for display
    return [(title, description) for title, description, _ in results]


# Step 3: Deploy with Gradio
def search_interface(query):
    results = smart_search(query)
    output = ""
    for title, description in results:
        output += f"**{title}**\n\n{description}\n\n---\n\n"
    return output


iface = gr.Interface(
    fn=search_interface,
    inputs="text",
    outputs="markdown",
    title="Smart Course Search Tool",
    description="Enter keywords to find the most relevant free courses on Analytics Vidhya."
)

# Launch the Gradio interface
iface.launch()
