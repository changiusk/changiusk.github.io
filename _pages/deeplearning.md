---
title: "Deep Learning"
layout: archive
permalink: /deeplearning/
author_profile: true
---

<div class="entries-{{ page.entries_layout }}">

{%- for post in site.categories['Deep Learning'] -%}
  {%- unless post.hidden -%}

  {% if post.id %}
    {% assign title = post.title | markdownify | remove: "<p>" | remove: "</p>" %}
  {% else %}
    {% assign title = post.title %}
  {% endif %}

  <div class="list__item">
    <article class="archive__item" itemscope itemtype="https://schema.org/CreativeWork">
    {{ forloop.rindex }}.
      {% if post.link %}
        <a href="{{ post.link }}">{{ title }}</a> <a href="{{ post.url | relative_url }}" rel="permalink"><i class="fas fa-link" aria-hidden="true" title="permalink"></i><span class="sr-only">Permalink</span></a>
      {% else %}
        <a href="{{ post.url | relative_url }}" rel="permalink">{{ title }}</a>
      {% endif %}

      {% if post.date %}
        <p class="page__meta"><i class="far fa-fw fa-calendar-alt" aria-hidden="true"></i> {{ post.date | date: "%B %d %Y" }}</p>
      {% endif %}
    </article>
  </div>
  {%- endunless -%}
{%- endfor -%}

</div>