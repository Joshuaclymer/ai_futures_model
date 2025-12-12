import { visit } from 'unist-util-visit';
import type { Root, Element, Content, Text } from 'hast';

type Parent = {
  type: string;
  children?: Content[];
};

function isElement(node: Content | Root | null | undefined): node is Element {
  return !!node && (node as any).type === 'element';
}

function isText(node: Content | Root | null | undefined): node is Text {
  return !!node && (node as any).type === 'text';
}

function getText(node: Content | Root | null | undefined): string {
  if (!node) return '';
  if (isText(node)) return String((node as Text).value || '');
  if (isElement(node)) return ((node as Element).children || []).map((c) => getText(c)).join('');
  const children = (node as any).children as Content[] | undefined;
  return (children || []).map((c) => getText(c)).join('');
}

function serializeMinimalHtml(node: any): string {
  if (!node) return '';
  if (node.type === 'text') {
    return node.value || '';
  }
  if (node.type === 'element') {
    const tag = node.tagName || 'span';
    const props = node.properties || {};
    const allowed: Record<string, string[]> = {
      a: ['href', 'id', 'name', 'rel', 'target', 'className', 'data-footnote-id'],
      span: ['id', 'className', 'style', 'data-footnote-id'],
      sup: [],
      sub: [],
      p: ['id', 'className', 'style'],
      div: ['id', 'className', 'style'],
      li: ['id', 'className', 'style']
    };
    const attrs = (allowed[tag] || ['id'])
      .map((k) => (props[k] != null ? ` ${k === 'className' ? 'class' : k}="${String(props[k])}"` : ''))
      .join('');
    const inner = (node.children || []).map(serializeMinimalHtml).join('');
    return `<${tag}${attrs}>${inner}</${tag}>`;
  }
  const children = Array.isArray((node as any).children) ? (node as any).children : [];
  return children.map(serializeMinimalHtml).join('');
}

function containsTargetElement(node: Element, key: string): boolean {
  const stack: (Element | Content)[] = [node];
  while (stack.length) {
    const cur = stack.pop() as Element | Content;
    if (isElement(cur)) {
      const props = (cur.properties || {}) as Record<string, any>;
      const id = typeof props.id === 'string' ? props.id : undefined;
      const name = typeof props.name === 'string' ? props.name : undefined;
      if (id === key || name === key) return true;
      const kids = (cur.children || []) as Content[];
      for (const k of kids) stack.push(k as any);
    }
  }
  return false;
}

export default function rehypeFootnotes() {
  return (tree: Root) => {
    // 1) Discover reference order from <a href="#ftntN"> regardless of label text
    const refOrder = new Map<string, number>();
    const refNodes: Array<{ parent: Parent; index: number; node: Element; target: string }> = [];
    let counter = 1;

    visit(tree as any, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number') return;
      if (node.tagName !== 'a') return;
      const props = (node.properties || {}) as Record<string, any>;
      const href = typeof props.href === 'string' ? props.href : '';
      if (!href || !href.startsWith('#')) return;
      const target = href.slice(1);
      if (!target) return;
      // Only treat links to bodies like #ftnt1 as references; skip backlinks like #ftnt_ref1
      if (!/^ftnt\d+$/.test(target)) return;
      if (!refOrder.has(target)) {
        refOrder.set(target, counter++);
      }
      refNodes.push({ parent, index, node, target });
    });

    if (refOrder.size === 0) return;

    // 2) Build map from original target -> body HTML by finding anchors with id/name equal to target
    const targets = new Set(refOrder.keys());
    const bodyHtmlByTarget = new Map<string, string>();
    const bodyTextByTarget = new Map<string, string>();
    const bodyBlockByTarget = new Map<string, Element>();

    visit(tree as any, 'element', (node: Element) => {
      if (!['p', 'li', 'div'].includes(node.tagName)) return;
      for (const key of targets) {
        if (!bodyHtmlByTarget.has(key) && containsTargetElement(node, key)) {
          const html = serializeMinimalHtml(node);
          const text = getText(node).trim();
          bodyHtmlByTarget.set(key, html);
          bodyTextByTarget.set(key, text);
          bodyBlockByTarget.set(key, node);
        }
      }
    });

    // 3) Rewrite reference anchors with stable IDs and inject hover popups
    for (const { parent, index, node, target } of refNodes) {
      const n = refOrder.get(target)!;
      const stableId = `fn-${n}`;

      // Update the existing anchor node in place
      const props = (node.properties || (node.properties = {} as any)) as Record<string, any>;
      props.href = `#${stableId}`;
      // Ensure className contains footnote-hover
      const prev = Array.isArray(props.className)
        ? props.className
        : (typeof props.className === 'string' && props.className.length > 0 ? [props.className] : []);
      props.className = Array.from(new Set([...(prev || []), 'footnote-hover']));
      props['data-footnote-id'] = stableId;

      // Create popup span as a sibling immediately after the anchor
      const popupText = (bodyTextByTarget.get(target) || '').slice(0, 300);
      const popup: Element = {
        type: 'element',
        tagName: 'span',
        properties: { className: 'footnote-popup' },
        children: [{ type: 'text', value: popupText } as any]
      } as any;

      (parent.children as Content[]).splice(index + 1, 0, popup as any);
    }

    // 4) Assign stable IDs to body blocks so anchor jumps work
    for (const [original, n] of refOrder.entries()) {
      const stableId = `fn-${n}`;
      const block = bodyBlockByTarget.get(original);
      if (block) {
        // Inject an empty span with id as the scroll target
        const anchor: Element = {
          type: 'element',
          tagName: 'span',
          properties: { id: stableId },
          children: []
        } as any;
        (block.children || (block.children = [] as any)).unshift(anchor as any);
      } else {
        // Fallback: set on any <a id|name=original>
        visit(tree as any, 'element', (node: Element) => {
          const props = (node.properties || {}) as Record<string, any>;
          const id = typeof props.id === 'string' ? props.id : undefined;
          const name = typeof props.name === 'string' ? props.name : undefined;
          const key = id || name;
          if (key === original) {
            (node.properties as any).id = stableId;
          }
        });
      }
    }
  };
}


