from setuptools import setup, find_packages

setup(
    name="ml-model-serving",
    version="1.0.0",
    description="Machine Learning Model Serving API with MLOps",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "aiofiles>=23.0.0",
        "asyncpg>=0.28.0",
        "prometheus-client>=0.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0