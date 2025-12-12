import { visit } from 'unist-util-visit';
import type { Root, Element, Content } from 'hast';

type Parent = {
  type: string;
  children?: Content[];
};

// Note: Comments (#cmnt) are now stripped at the raw HTML level in googleDocToMarkdown.ts
// This file only handles footnotes (#ftnt)

function hasFtntAnchor(node: Element): boolean {
  const stack: Array<Element | Content> = [node];
  while (stack.length) {
    const cur = stack.pop() as Element | Content;
    if ((cur as any)?.type === 'element' && (cur as any).tagName === 'a') {
      const href = (cur as any).properties?.href as string | undefined;
      if (typeof href === 'string' && href.startsWith('#ftnt')) return true;
    }
    const kids = ((cur as any)?.children as Array<Element | Content>) || [];
    for (const k of kids) stack.push(k);
  }
  return false;
}

function containsFtntIdOrName(node: Element): boolean {
  const stack: Array<Element | Content> = [node];
  while (stack.length) {
    const cur = stack.pop() as Element | Content;
    if ((cur as any)?.type === 'element') {
      const id = (cur as any).properties?.id as string | undefined;
      const name = (cur as any).properties?.name as string | undefined;
      const href = (cur as any).properties?.href as string | undefined;
      if ((id && /^ftnt/.test(id)) || (name && /^ftnt/.test(name))) return true;
      if (href && /^#ftnt/.test(href)) return true;
    }
    const kids = ((cur as any)?.children as Array<Element | Content>) || [];
    for (const k of kids) stack.push(k);
  }
  return false;
}

export default function rehypeStripFootnotes() {
  return (tree: Root) => {
    // 0) Convert full URLs with anchors to just anchor links (same-page navigation)
    visit(tree as any, 'element', (node: Element) => {
      if (node.tagName === 'a' && node.properties?.href) {
        const href = node.properties.href as string;
        if (href.startsWith('https://ai-rates-calculator.vercel.app/#')) {
          node.properties.href = href.replace('https://ai-rates-calculator.vercel.app/', '');
        }
      }
    });

    // 1) Remove inline footnote references
    visit(tree as any, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number') return;

      // Remove <sup> that wraps a footnote link
      if (node.tagName === 'sup' && hasFtntAnchor(node)) {
        (parent.children as any[]).splice(index, 1);
        return;
      }

      // Remove bare <a href="#ftntN">
      if (node.tagName === 'a') {
        const href = (node.properties?.href as string) || '';
        if (href.startsWith('#ftnt')) {
          (parent.children as any[]).splice(index, 1);
          return;
        }
      }
    });

    // 2) Remove footnote body blocks
    const blockTags = new Set(['p', 'li', 'div', 'section', 'ol', 'ul']);
    visit(tree as any, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number') return;
      if (!blockTags.has(node.tagName)) return;
      if (containsFtntIdOrName(node)) {
        (parent.children as any[]).splice(index, 1);
        return;
      }
    });
  };
}


