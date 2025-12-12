import fs from 'fs';
import { unified } from 'unified';
import rehypeParse from 'rehype-parse';
import rehypeRemark from 'rehype-remark';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkStringify from 'remark-stringify';
import rehypeGoogleStylesToSemantic from '@/utils/rehypeGoogleStylesToSemantic';

function serializeMinimalHtml(node: any): string {
  if (!node) return '';
  if (node.type === 'text') return node.value || '';
  if (node.type === 'element') {
    const tag = node.tagName || 'span';
    const children = Array.isArray(node.children) ? node.children : [];
    const inner = children.map(serializeMinimalHtml).join('');
    return `<${tag}>${inner}</${tag}>`;
  }
  const children = Array.isArray(node.children) ? node.children : [];
  return children.map(serializeMinimalHtml).join('');
}

async function main() {
  const html = fs.readFileSync('tmp_inlined.html', 'utf8');
  const file = await unified()
    .use(rehypeParse, { fragment: true })
    .use(rehypeGoogleStylesToSemantic)
    .use(rehypeRemark as any, {
      handlers: {
        sup: (_state: any, node: any) => ({ type: 'html', value: serializeMinimalHtml(node) }),
        sub: (_state: any, node: any) => ({ type: 'html', value: serializeMinimalHtml(node) }),
      },
    } as any)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkStringify as any, { fences: true, bullet: '-', allowDangerousHtml: true } as any)
    .process(html);
  const md = String(file);
  console.log('MD length:', md.length);
  console.log(md.slice(0, 1000));
  const match = md.match(/C<sub>.*?<\/sub>/i);
  console.log('C sub match:', match ? match[0] : 'none');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


