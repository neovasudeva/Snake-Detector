# Overview
This application is used to detect snakes given an image. It's intent was to help detect snakes around my house, but because 
I don't currently have the resources to set that up, I have (for now) deployed a small web app to demonstrate a proof of concept.

# Model
For the model, I used FAIR's Detectron2 and used transfer learning to train a Mask-RCNN model. Images were scraped from the web
using Selenium and labeled using the LabelMe tool. Once labeled, the images were fed to the model and trained on Google Colab.
<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/neovasudeva/Snake-Detector/dev/images/im1.jpg" />
</p>
<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/neovasudeva/Snake-Detector/dev/images/im3.jpg" />
</p>
<br>
After asking friends to try the model out, I've realized this model is only good with finding snakes far from the camera and in the outdoors 
(after all, that's what it was meant for). The model is terrible at identifying snakes in closeup images and even worse at finding 
snakes that don't have some pattern on its skin.

# Web application
I like a modular style of building applications, so I decided to employ a microservices approach to building this application.
I separated the application into logical partitions (frontend, backend, api, database) and had each service communicate with each 
other over REST endpoints.

The API service is supposed to serve as the API gateway (because NGINX Plus and AWS Lambda is not free). I was going to make it non-blocking,
but because Python has only recently supported asynchronous I/O (aiohttp) and is constantly changing it with every new version
of Python, I decided to wait until it settles.

# Deployment
The application was Dockerized and orchestrated with Kubernetes. Because each service was stateless (except the database), everything
was put into a deployment and can be easily scaled up or down. The database is a stateful service, but because demand is not high, I have
just made a Deployment with a single replica (and persistent volumes of course). 

GCP, AWS, and Azure Kubernetes services were very expensive for this hobby web application. I decided to host on DigitalOcean because
they didn't slap on a maintenance fee (looking at you GCP) for master services. Following DOKS's guide, I also added SSL/TLS certificates 
because Chrome hates website that don't have them.
