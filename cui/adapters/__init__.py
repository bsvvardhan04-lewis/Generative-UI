from . import adaptive_card, html, markdown

RENDERERS = {"markdown": markdown.render, "card": adaptive_card.render, "html": html.render}
