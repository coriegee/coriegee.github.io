---
layout: inner
title: About
permalink: /about/
---
<html>
<head>
  <style>
    h1 {
      font-size: 48px;
    }
	
	p {
	  font-size: 14px;
	  font-family: 'Open Sans', sans-serif;
	}
	
	.image-container {
      display: flex;
      align-items: center;
    }

    .image-container .text-block {
      margin-right: 20px;
    }

    .image-container img {
      width: 200px;
      height: auto;
      border-radius: 50%;
    }
    }
  </style>
  <title>Corie Gee | Data Scientist</title>
</head>
<body>
	<br>
	<br>
	<br>
	<div align="center">
	  <h1>Corie Gee | Data Scientist</h1>
	</div>
	<br>
	<br>
	<div class="image-container" style="display: flex; flex-direction: column; align-items: center;">
	  <img src="/img/Airbrush-AIGC-5.png" alt="Image" style="max-width: 100%; height: auto; margin-bottom: 20px;">
	  <div class="text-block">
		<p>Hey there! I'm a results-driven data scientist with over 4 years of experience in statistical analysis, machine learning, and data visualization. My journey has been fueled by a passion for turning complex datasets into actionable insights that fuel business growth.
		  <br><br>With a strong foundation in Python, SQL, Tableau, and Power BI, I've designed and deployed dashboards that analyze trends and track various metrics. My expertise in developing statistical measures and data visualizations has been instrumental in identifying emerging issues and cost-saving opportunities at various companies.
		  <br><br>My work doesn't stop there – I've built predictive models for business decision-making, developed real-time monitoring dashboards, and contributed to digital transformation efforts. I'm also certified in AWS Cloud, Tableau, and Alteryx, demonstrating my commitment to staying current in the ever-evolving data landscape.
		  <br><br>I bring a blend of technical skills and a passion for delivering impactful solutions through data-driven insights. Let's connect and explore how I can contribute to your data science needs!
		</p>
	  </div>
	</div>
	<br>
	<br>
	<div align="center">
		<a href="/">Check out my project portfolio</a>
	</div>
	<div align="center" class="hero-buttons">

            {% if site.twitter_username != '' %}
              <a href="https://www.linkedin.com/in/{{ site.twitter_username }}"><button class="btn btn-default btn-lg"><i class="fa fa-twitter fa-lg"></i>LinkedIn</button></a>
            {% endif %}
			{% if site.linkedin_username != '' %}
              <a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"><button class="btn btn-default btn-lg"><i class="fa fa-linkedin fa-lg"></i>LinkedIn</button></a>
            {% endif %}
            {% if site.github_username != '' %}
              <a href="https://github.com/{{ site.github_username }}"><button class="btn btn-default btn-lg"><i class="fa fa-github fa-lg"></i>Github</button></a>
            {% endif %}
            {% if site.medium_username != '' %}
              <a href="https://{{ site.medium_username }}"><button class="btn btn-default btn-lg"><i class="fa fa-medium fa-lg"></i>Medium</button></a>
            {% endif %}
    </div>
</body>
</html>


