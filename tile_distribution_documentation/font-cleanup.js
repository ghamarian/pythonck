// Font cleanup script to run on page load
// This ensures consistent fonts even if some styles slip through

document.addEventListener('DOMContentLoaded', function() {
    // Define our standard fonts
    const SANS_SERIF = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
    const MONOSPACE = '"SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace';
    
    // Remove all serif fonts
    const allElements = document.querySelectorAll('*');
    allElements.forEach(el => {
        const computedStyle = window.getComputedStyle(el);
        const fontFamily = computedStyle.fontFamily;
        
        // Check if element has serif font
        if (fontFamily && (
            fontFamily.includes('serif') && !fontFamily.includes('sans-serif') ||
            fontFamily.includes('Times') ||
            fontFamily.includes('Georgia') ||
            fontFamily.includes('Cambria')
        )) {
            el.style.fontFamily = SANS_SERIF;
        }
        
        // Check for unwanted bold
        const fontWeight = computedStyle.fontWeight;
        const tagName = el.tagName.toLowerCase();
        const isHeading = /^h[1-6]$/.test(tagName);
        const isTitle = el.classList.contains('title') || el.classList.contains('heading');
        
        // Remove bold from non-headings
        if (!isHeading && !isTitle && (fontWeight === 'bold' || fontWeight === '700' || fontWeight === '600')) {
            // Check if it's in a paragraph or list
            const parent = el.parentElement;
            if (parent && (parent.tagName === 'P' || parent.tagName === 'LI' || parent.tagName === 'TD')) {
                el.style.fontWeight = '500'; // medium weight for emphasis
            }
        }
    });
    
    // Fix SVG text elements
    const svgTexts = document.querySelectorAll('svg text');
    svgTexts.forEach(text => {
        if (!text.style.fontFamily || text.style.fontFamily.includes('serif')) {
            text.style.fontFamily = SANS_SERIF;
        }
        
        // Check if it's not a title or heading
        if (!text.classList.contains('title') && !text.classList.contains('heading')) {
            const currentWeight = text.style.fontWeight;
            if (currentWeight === 'bold' || currentWeight === '700' || currentWeight === '600') {
                text.style.fontWeight = '400'; // normal weight
            }
        }
    });
    
    // Fix code elements
    const codeElements = document.querySelectorAll('code, pre, .sourceCode, .code');
    codeElements.forEach(code => {
        code.style.fontFamily = MONOSPACE;
        code.style.fontWeight = '400'; // code should never be bold
    });
    
    // Clean up Quarto specific elements
    const quartoMeta = document.querySelectorAll('.quarto-title-meta-heading, .quarto-title-meta-contents');
    quartoMeta.forEach(el => {
        el.style.fontFamily = SANS_SERIF;
        el.style.fontWeight = '400';
    });
    
    // Remove bold from navigation
    const navElements = document.querySelectorAll('.sidebar-item-text, .toc-item a, .navbar-nav .nav-link');
    navElements.forEach(el => {
        el.style.fontWeight = '400';
    });
    
    console.log('Font cleanup completed');
});