# Overview
This application is used to detect snakes given an image. It's intent was to help detect snakes around my house, but because 
I don't currently have the resources to set that up, I have (for now) deployed a small web app to demonstrate a proof of concept.

# Model
For the model, I used FAIR's Detectron2 and used transfer learning to train a Mask-RCNN model. Images were scraped from the web
using Selenium and labeled using the LabelMe tool. Once labeled, the images were fed to the model and trained on Google Colab.
![GitHub Logo](/images/0.png)
Format: ![Alt Text](image1)
![GitHub Logo](/images/2.png)
Format: ![Alt Text](image2)

# Web application
I like a modular style of building applications, so I decided to employ a microservices approach to building this application.
I separated the application into logical partitions (frontend, backend, api, database) and had each service communicate with each 
other over REST endpoints.

The api service is supposed to serve as the API gateway (because NGINX Plus is not free). I was going to make it non-blocking,
but because Python has only recently supported asynchronous I/O (aiohttp) and is constantly changing it with every new version
of Python, I decided to wait until it settles.

# Deployment
The application was Dockerized and orchestrated with Kubernetes. Because each service was stateless (except the database), everything
was put into a deployment and can be easily scaled up or down. The database is a stateful service, but because demand is not high, I have
just made a deployment with a single replica for it (along with persistent volumes). 
