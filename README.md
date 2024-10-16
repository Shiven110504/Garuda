# Garuda Dashboard

[![Garuda Dashboard Demo](https://img.youtube.com/vi/Whb40BVR0S8/0.jpg)](https://www.youtube.com/watch?v=Whb40BVR0S8)

## Introduction

Natural disasters do more than just destroy property—they disrupt lives, tear apart communities, and hinder our progress toward a sustainable future. One of our team members from Rice University experienced this firsthand during a recent hurricane in Houston. Trees were uprooted, infrastructure was destroyed, and delayed response times put countless lives at risk.

### Emotional Impact
The chaos and helplessness during such events are overwhelming.

### Urgency for Change
We recognized the need for swift damage assessment to aid authorities in locating those in need and deploying appropriate services.

### Sustainability Concerns
Rebuilding efforts often use non-eco-friendly methods, leading to significant carbon footprints.

Inspired by these challenges, we aim to leverage AI, computer vision, and peer networks to provide rapid, actionable damage assessments. Our AI assistant can detect people in distress and deliver crucial information swiftly, bridging the gap between disaster and recovery.

## What it Does

**Garuda Dashboard** offers a comprehensive view of current, upcoming, and past disasters across the country:

- **Live Dashboard**: Displays a heatmap of affected areas updated via a peer-to-peer network.
- **Drones Damage Analysis**: Deploy drones to survey and mark damaged neighborhoods using the Llava Vision-Language Model and generate reports for the Recovery Team.
- **Detailed Reporting**: Reports have annotations to classify damage types [tree, road, roof, water], human rescue needs, site accessibility [Can response team get to the site by land], and suggest equipment dispatch [Cranes, Ambulance, Fire Control].
- **Drowning Alert**: The drone footage can detect when it identifies a drowning subject and immediately call rescue teams.
- **AI-Generated Summary**: Reports on past disasters include recovery costs, carbon footprint, and total asset/life damage.

## How We Built It

- **Front End**: Developed with Next.js for an intuitive user interface tailored for emergency use.
- **Data Integration**: Utilized Google Maps API for heatmaps and energy-efficient routing.
- **Real-Time Updates**: Custom Flask API records hot zones when users upload disaster videos.
- **AI Models**: Employed MSNet for real-time damage assessment on GPUs and Llava VLM for detailed video analysis.
- **Secure Storage**: Images and videos stored on Firebase database.

## Challenges We Faced

- **Model Integration**: Adapting MSNet with outdated dependencies required deep understanding of technical papers.
- **VLM Setup**: Implementing Llava VLM was challenging due to lack of prior experience.
- **Efficiency Issues**: Running models on personal computers led to inefficiencies.

## Accomplishments We're Proud Of

- **Technical Skills**: Mastered API integration, technical paper analysis, and new technologies like VLMs.
- **Innovative Impact**: Combined emerging technologies for disaster detection and recovery measures.
- **Complex Integration**: Successfully merged backend, frontend, and GPU components under time constraints.

## What We Learned

- Expanded full-stack development skills and explored new AI models.
- Realized the potential of coding experience in tackling real-world problems with interdisciplinary solutions.
- Balanced MVP features with user needs throughout development.

## What's Next for Garuda

- **Drone Integration**: Enable drones to autonomously call EMS services and deploy life-saving equipment.
- **Collaboration with EMS**: Partner with emergency services for widespread national and global adoption.
- **Broader Impact**: Expand software capabilities to address various natural disasters beyond hurricanes.
