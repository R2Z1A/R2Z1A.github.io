/**
 * Fix tags page when there are no tags
 * Ensures the page title and proper content are displayed
 */

hexo.extend.filter.register('after_render:html', function(str, data) {
  // Only process tags page
  if (!data || !data.path || !data.path.includes('tags/')) {
    return str;
  }

  // Check if this is an empty tags page (missing page-title-header)
  if (str.includes('page-template-container') && !str.includes('page-title-header')) {
    // Get the page title
    let title = '标签';
    if (data.page && data.page.title) {
      title = data.page.title;
    }

    // Insert the title and empty state content into the page-template-content
    const replacement = `
    <div class="page-template-container">
        <h1 class="page-title-header">${title}</h1>
        <div class="page-template-content markdown-body">
            <div class="tagcloud-content">
                <p style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    暂无标签
                </p>
            </div>
        </div>
        <div class="page-template-comments">
        </div>
    </div>
    `;

    // Replace the empty container
    str = str.replace(
        /<div class="page-template-container">\s*<div class="page-template-content markdown-body">\s*<\/div>\s*<div class="page-template-comments">\s*<\/div>\s*<\/div>/,
        replacement
    );
  }

  return str;
});
