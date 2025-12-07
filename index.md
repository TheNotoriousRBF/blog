---
layout: home
title: "The Notorious RBF Blog"
---

# Welcome to my blog

This is my pseudonymous blog about biostatistics and oncology  
Here you’ll find dicussion, code snippets, thinking out loud, and critique of ideas

---

## Latest Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>  
      - <small>{{ post.date | date: "%B %d, %Y" }}</small>  
      {% if post.categories %}
        <em> (Categories: {{ post.categories | join: ", " }})</em>
      {% endif %}
    </li>
  {% endfor %}
</ul>

---

## Categories

<ul>
  {% assign all_categories = site.posts | map: "categories" | flatten | uniq %}
  {% for category in all_categories %}
    <li>{{ category }}</li>
  {% endfor %}
</ul>
