from web.style import CSS, _one_html_block


def test_injected_css_has_no_blank_lines():
    # A blank line would end the Markdown HTML block and print the rest of the CSS as text.
    out = _one_html_block(CSS)
    assert "\n\n" not in out and out.strip() != ""
    assert ".le-tiles" in out and "--accent: #b8410e" in out
