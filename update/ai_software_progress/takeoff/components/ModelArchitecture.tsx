'use client';

import ModelDiagram from '@/svgs/model-diagram-semantic.svg';
import { useCallback, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react';
import explanationsHtml from '@/public/explanations.html';
import { useParameterHover, PARAMETER_TO_SVG_NODES } from './ParameterHoverContext';

// Override mapping: SVG node label -> heading text to scroll to
// Keys are normalized labels (whitespace collapsed to single spaces)
export const SVG_LABEL_TO_HEADING: Record<string, string> = {
  'Automation compute': 'Compute forecasts',
  'Experiment compute': 'Compute forecasts',
  'Training compute': 'Compute forecasts',
  'Human coding labor': 'Aggregate coding labor',
  'Software research effort': 'Research effort and AI software R&D uplift',
  'Automated Coder Time Horizon': 'Time horizon and the Automated Coder milestone',
  'Effective Compute': 'Modeling effective compute, EL version',
};

type SectionEntry = {
  label: string;
  x: number | null;
};

const normalizeLabel = (input: string | null | undefined) => {
  if (!input) {
    return null;
  }
  const normalized = input.replace(/\s+/g, ' ').trim();
  return normalized.length > 0 ? normalized : null;
};

const normalizeHeadingText = (input: string | null | undefined) => {
  if (!input) {
    return null;
  }
  const withoutPrefix = input.replace(/^[\d\s.:()-]+/, '');
  return normalizeLabel(withoutPrefix);
};

const buildMatchKey = (input: string | null | undefined) => {
  const normalized = normalizeLabel(input);
  if (!normalized) {
    return null;
  }
  return normalized.toLowerCase().replace(/[^a-z0-9]+/g, '');
};

const parseXCoordinate = (element: Element | null) => {
  if (!element) {
    return null;
  }
  const rawValue = element.getAttribute('x');
  if (!rawValue) {
    return null;
  }
  const parsed = Number.parseFloat(rawValue);
  return Number.isNaN(parsed) ? null : parsed;
};

const findNearestSectionLabel = (sections: SectionEntry[], nodeX: number | null) => {
  if (sections.length === 0) {
    return null;
  }
  if (nodeX == null) {
    return sections[0].label;
  }
  const nearest = sections.reduce<SectionEntry>((best, candidate) => {
    if (candidate.x == null) {
      return best;
    }
    if (best.x == null) {
      return candidate;
    }
    const bestDistance = Math.abs(best.x - nodeX);
    const candidateDistance = Math.abs(candidate.x - nodeX);
    return candidateDistance < bestDistance ? candidate : best;
  }, sections[0]);

  return nearest.label;
};

const findForeignObject = (element: Element | null) => {
  let current: Element | null = element;
  while (current && current.tagName.toLowerCase() !== 'foreignobject') {
    current = current.parentElement;
  }

  return current;
};

type DiagramLabelElement = HTMLElement | SVGTextElement;

const isSvgTextElement = (element: Element): element is SVGTextElement =>
  element.tagName.toLowerCase() === 'text';

const isTspanElement = (element: Element): element is SVGTSpanElement =>
  element.tagName.toLowerCase() === 'tspan';

const isParagraphSection = (element: HTMLElement) =>
  element.style.fontStyle === 'italic' && element.style.fontSize === '48px';

const TEXT_FONT_THRESHOLD = 12;

const shouldIncludeSvgTextElement = (element: SVGTextElement) => {
  const label = normalizeLabel(element.textContent);
  if (!label) {
    return false;
  }
  const fontSizeAttr = element.getAttribute('font-size');
  const fontSize = fontSizeAttr ? Number.parseFloat(fontSizeAttr) : Number.NaN;
  if (!Number.isNaN(fontSize) && fontSize < TEXT_FONT_THRESHOLD) {
    return false;
  }
  return true;
};

const getLabelElements = (svg: SVGElement): DiagramLabelElement[] => {
  const paragraphElements = Array.from(svg.querySelectorAll('foreignObject p')) as HTMLElement[];
  const textElements = Array.from(svg.querySelectorAll('text')).filter((text): text is SVGTextElement =>
    isSvgTextElement(text) && shouldIncludeSvgTextElement(text),
  );
  return [...paragraphElements, ...textElements];
};

const getElementXCoordinate = (element: DiagramLabelElement): number | null => {
  if (element instanceof SVGGraphicsElement) {
    try {
      const bbox = element.getBBox();
      return Number.isFinite(bbox.x) ? bbox.x : null;
    } catch {
      return null;
    }
  }
  const foreignObject = findForeignObject(element);
  return parseXCoordinate(foreignObject);
};

const ensureOriginalColor = (element: DiagramLabelElement) => {
  if (isSvgTextElement(element)) {
    if (!element.hasAttribute('data-original-fill')) {
      const fill = element.getAttribute('fill') ?? '#1E1E1E';
      element.setAttribute('data-original-fill', fill);
    }
  } else if (!element.hasAttribute('data-original-color')) {
    const computedStyle = window.getComputedStyle(element);
    element.setAttribute('data-original-color', computedStyle.color);
  }
};

const restoreOriginalColor = (element: DiagramLabelElement) => {
  if (isSvgTextElement(element)) {
    const originalFill = element.getAttribute('data-original-fill');
    if (originalFill) {
      element.setAttribute('fill', originalFill);
    }
  } else {
    const originalColor = element.getAttribute('data-original-color');
    if (originalColor) {
      element.style.color = originalColor;
    }
  }
};

const setTextHighlightColor = (element: DiagramLabelElement, color: string) => {
  if (isSvgTextElement(element)) {
    element.setAttribute('fill', color);
  } else {
    element.style.color = color;
  }
};

const findLabelElement = (target: EventTarget | null): DiagramLabelElement | null => {
  let current: Element | null = null;
  if (target instanceof Element) {
    current = target;
  } else if (target instanceof Node) {
    current = target.parentElement;
  }

  while (current) {
    const tagName = current.tagName.toLowerCase();
    if (tagName === 'p' || tagName === 'text') {
      return current as DiagramLabelElement;
    }
    if (tagName === 'tspan') {
      current = current.parentElement;
      continue;
    }
    current = current.parentElement;
  }

  return null;
};

type PointerPosition = {
  x: number;
  y: number;
};

const MAX_POINTER_DISTANCE = 140;

const getNearestLabelElement = (svg: SVGElement, pointer: PointerPosition): DiagramLabelElement | null => {
  const labelElements = getLabelElements(svg);
  let closestElement: DiagramLabelElement | null = null;
  let closestDistance = Number.POSITIVE_INFINITY;

  labelElements.forEach((element) => {
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) {
      return;
    }
    const clampedX = Math.min(Math.max(pointer.x, rect.left), rect.right);
    const clampedY = Math.min(Math.max(pointer.y, rect.top), rect.bottom);
    const dx = pointer.x - clampedX;
    const dy = pointer.y - clampedY;
    const distance = Math.hypot(dx, dy);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestElement = element;
    }
  });

  if (closestDistance <= MAX_POINTER_DISTANCE) {
    return closestElement;
  }
  return null;
};

const buildSectionEntries = (svg: SVGElement): SectionEntry[] =>
  Array.from(svg.querySelectorAll('foreignObject')).reduce<SectionEntry[]>(
    (acc, foreignObject) => {
      const paragraph = foreignObject.querySelector('p');
      if (!paragraph) {
        return acc;
      }
      const label = normalizeLabel(paragraph.textContent);
      if (!label) {
        return acc;
      }

      const paragraphElement = paragraph as HTMLElement;
      const isSection =
        paragraphElement.style.fontStyle === 'italic' &&
        paragraphElement.style.fontSize === '48px';

      if (!isSection) {
        return acc;
      }

      acc.push({
        label,
        x: parseXCoordinate(foreignObject),
      });
      return acc;
    },
    [],
  );

const hasValidTarget = (label: string | null, headingLookup: Map<string, HTMLElement>): boolean => {
  if (!label) {
    return false;
  }

  // Check override mapping
  const overrideHeading = SVG_LABEL_TO_HEADING[label];
  const key = buildMatchKey(overrideHeading ?? label);
  if (!key) {
    return false;
  }

  let target = headingLookup.get(key);
  if (!target) {
    for (const [storedKey, element] of headingLookup.entries()) {
      if (storedKey.includes(key) || key.includes(storedKey)) {
        target = element;
        break;
      }
    }
  }

  return target != null;
};

type ModelArchitectureProps = {
  modelDescriptionGDocPortionMarkdown?: React.ReactNode;
  introContent?: ReactNode;
  scrollContainerRef?: React.RefObject<HTMLDivElement | null>;
};

export default function ModelArchitecture({ modelDescriptionGDocPortionMarkdown, introContent, scrollContainerRef }: ModelArchitectureProps) {
  const explanationContainerRef = useRef<HTMLDivElement | null>(null);
  const headingLookupRef = useRef<Map<string, HTMLElement>>(new Map());
  const svgContainerRef = useRef<HTMLDivElement | null>(null);
  const mobileSvgContainerRef = useRef<HTMLDivElement | null>(null);
  const mobileContentRef = useRef<HTMLDivElement | null>(null);
  const [isHoveringSvg, setIsHoveringSvg] = useState(false);
  const [isMobileDiagramVisible, setIsMobileDiagramVisible] = useState(false);
  const { hoveredParameter } = useParameterHover();

  // Track scroll position to show/hide mobile diagram
  useEffect(() => {
    const SCROLL_THRESHOLD = 500; // Show diagram after scrolling 500px into the essay

    const handleScroll = () => {
      const contentEl = mobileContentRef.current;
      const scrollContainer = scrollContainerRef?.current;
      if (!contentEl) return;

      // Get the scroll position relative to the content
      // Use the scroll container if provided, otherwise fall back to viewport
      if (scrollContainer) {
        const containerRect = scrollContainer.getBoundingClientRect();
        const contentRect = contentEl.getBoundingClientRect();
        // Calculate how far the content has scrolled up relative to the scroll container
        const scrolledIntoContent = Math.max(0, containerRect.top - contentRect.top);
        setIsMobileDiagramVisible(scrolledIntoContent >= SCROLL_THRESHOLD);
      } else {
        // Fallback to viewport-based calculation
        const rect = contentEl.getBoundingClientRect();
        const scrolledIntoContent = Math.max(0, -rect.top);
        setIsMobileDiagramVisible(scrolledIntoContent >= SCROLL_THRESHOLD);
      }
    };

    // Check initial scroll position
    handleScroll();

    // Listen on the scroll container if provided, otherwise window
    const scrollTarget = scrollContainerRef?.current ?? window;
    scrollTarget.addEventListener('scroll', handleScroll, { passive: true });
    return () => scrollTarget.removeEventListener('scroll', handleScroll);
  }, [scrollContainerRef]);

  useEffect(() => {
    const container = explanationContainerRef.current;
    const nextLookup = new Map<string, HTMLElement>();

    if (container) {
      const headings = container.querySelectorAll('h2, h3');
      headings.forEach((heading) => {
        const normalizedHeading = normalizeHeadingText(heading.textContent);
        const key = buildMatchKey(normalizedHeading);
        if (!key || nextLookup.has(key)) {
          return;
        }
        const element = heading as HTMLElement;
        if (!element.id) {
          element.id = `section-${key}`;
        }
        if (!element.hasAttribute('tabindex')) {
          element.setAttribute('tabindex', '-1');
        }
        element.style.scrollMarginTop = '30rem';
        nextLookup.set(key, element);
      });
    }

    headingLookupRef.current = nextLookup;
  }, []);

  const scrollToHeading = useCallback((label: string | null) => {
    if (!label) {
      return;
    }

    // Check for manual override first
    const overrideHeading = SVG_LABEL_TO_HEADING[label];
    const key = buildMatchKey(overrideHeading ?? label);
    if (!key) {
      return;
    }

    const lookup = headingLookupRef.current;
    let target = lookup.get(key);
    if (!target) {
      for (const [storedKey, element] of lookup.entries()) {
        if (storedKey.includes(key) || key.includes(storedKey)) {
          target = element;
          break;
        }
      }
    }

    if (!target) {
      return;
    }

    target.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }, []);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const handleMouseEnter = () => {
      setIsHoveringSvg(true);
    };

    const handleMouseLeave = () => {
      setIsHoveringSvg(false);
    };

    svg.addEventListener('mouseenter', handleMouseEnter);
    svg.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      svg.removeEventListener('mouseenter', handleMouseEnter);
      svg.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const headingLookup = headingLookupRef.current;
    const sectionEntries = buildSectionEntries(svg);
    const labelElements = getLabelElements(svg);

    labelElements.forEach((element) => {
      const label = normalizeLabel(element.textContent);

      if (!label) {
        element.style.textDecoration = '';
        element.style.textDecorationStyle = '';
        element.style.textDecorationColor = '';
        element.style.cursor = '';
        return;
      }

      ensureOriginalColor(element);

      const isSection =
        element instanceof HTMLElement ? isParagraphSection(element) : false;

      const nodeX = getElementXCoordinate(element);
      const sectionLabel = isSection ? label : findNearestSectionLabel(sectionEntries, nodeX);
      const nodeLabel = isSection ? null : label;
      const targetLabel = nodeLabel ?? sectionLabel;

      const isValidLink = hasValidTarget(targetLabel, headingLookup) && !isSection;

      if (isHoveringSvg && isValidLink) {
        element.style.textDecorationStyle = 'dotted';
        element.style.textDecorationColor = '#2A623D';
        element.style.textDecorationThickness = '3px';
        element.style.cursor = 'pointer';
      } else {
        element.style.textDecoration = '';
        element.style.textDecorationStyle = '';
        element.style.textDecorationColor = '';
        element.style.cursor = '';
        restoreOriginalColor(element);
      }
    });
  }, [isHoveringSvg]);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const highlightLabels = hoveredParameter ? PARAMETER_TO_SVG_NODES[hoveredParameter] : null;

    const labelElements = getLabelElements(svg);
    labelElements.forEach((element) => {
      const label = normalizeLabel(element.textContent);

      if (!label) {
        return;
      }

      ensureOriginalColor(element);

      const shouldHighlight = highlightLabels?.includes(label) ?? false;

      if (shouldHighlight) {
        if (element instanceof HTMLElement) {
          element.style.color = '#8b0000';
          element.style.fontWeight = 'bold';
          element.style.padding = '2px 4px';
          element.style.borderRadius = '3px';
          element.style.transition = 'all 0.15s ease-in-out';
        } else {
          element.style.fontWeight = 'bold';
          setTextHighlightColor(element, '#8b0000');
        }
      } else {
        if (element instanceof HTMLElement) {
          element.style.backgroundColor = '';
          element.style.fontWeight = '';
          element.style.padding = '';
          element.style.borderRadius = '';
          restoreOriginalColor(element);
        } else {
          element.style.fontWeight = '';
          restoreOriginalColor(element);
        }
      }
    });
  }, [hoveredParameter]);

  const navigateFromSvgTarget = useCallback((eventTarget: EventTarget | null, svg: SVGElement, pointer?: PointerPosition) => {
    // First, check if we clicked on a group with a data-label
    let element = eventTarget as Element | null;
    while (element && element !== svg) {
      const dataLabel = element.getAttribute('data-label');
      if (dataLabel) {
        scrollToHeading(dataLabel);
        console.log('model-architecture-click', { box: dataLabel });
        return;
      }
      element = element.parentElement;
    }

    // Fall back to original label element logic
    let labelElement = findLabelElement(eventTarget);
    if (!labelElement && pointer) {
      labelElement = getNearestLabelElement(svg, pointer);
    }
    if (!labelElement) {
      return;
    }

    const label = normalizeLabel(labelElement.textContent);
    if (!label) {
      return;
    }

    const nodeX = getElementXCoordinate(labelElement);

    const isSection =
      labelElement instanceof HTMLElement ? isParagraphSection(labelElement) : false;

    const sectionEntries = buildSectionEntries(svg);

    const sectionLabel = isSection ? label : findNearestSectionLabel(sectionEntries, nodeX);

    const nodeLabel = isSection ? null : label;
    scrollToHeading(nodeLabel ?? sectionLabel);

    console.log('model-architecture-click', {
      node: nodeLabel ?? undefined,
      section: sectionLabel ?? undefined,
    });
  }, [scrollToHeading]);

  const handleSvgClick = useCallback((event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();

    const svg = event.currentTarget as SVGElement | null;
    if (!svg) {
      return;
    }

    navigateFromSvgTarget(event.target, svg, { x: event.clientX, y: event.clientY });
  }, [navigateFromSvgTarget]);

  const handleContainerClick = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const svg = svgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }
    navigateFromSvgTarget(event.target, svg, { x: event.clientX, y: event.clientY });
  }, [navigateFromSvgTarget]);

  useEffect(() => {
    const svg = svgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }

    svg.addEventListener('click', handleSvgClick);
    return () => {
      svg.removeEventListener('click', handleSvgClick);
    };
  }, [handleSvgClick]);

  // Click handler for mobile SVG
  useEffect(() => {
    const svg = mobileSvgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }

    svg.addEventListener('click', handleSvgClick);
    return () => {
      svg.removeEventListener('click', handleSvgClick);
    };
  }, [handleSvgClick]);

  // Make SVG responsive by removing fixed dimensions and setup hover effects
  useEffect(() => {
    const setupSvg = (container: HTMLDivElement | null, isMobile: boolean = false) => {
      const svg = container?.querySelector('svg');
      if (!svg) return;
      
      // Store original viewBox or create one from width/height
      const width = svg.getAttribute('width');
      const height = svg.getAttribute('height');
      if (!svg.getAttribute('viewBox') && width && height) {
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      }
      
      // Remove fixed dimensions so CSS can control size
      svg.removeAttribute('width');
      svg.removeAttribute('height');
      
      if (isMobile) {
        // For mobile: constrain by height, let width scale proportionally
        svg.style.maxHeight = '100%';
        svg.style.width = 'auto';
        svg.style.height = 'auto';
      } else {
        // For desktop: fill width, let height scale proportionally
        svg.style.width = '100%';
        svg.style.height = 'auto';
      }

      // Setup hover effects for diagram boxes using data attributes
      const diagramBoxes = svg.querySelectorAll('[data-label]');
      
      diagramBoxes.forEach((box) => {
        const bgRect = box.querySelector('[data-box-bg]') as SVGRectElement | null;
        const textElements = box.querySelectorAll('text, tspan');
        
        // Store original fill values
        const originalFills = new Map<Element, string | null>();
        const originalOpacities = new Map<Element, string | null>();
        textElements.forEach((el) => {
          originalFills.set(el, el.getAttribute('fill'));
          originalOpacities.set(el, el.getAttribute('fill-opacity'));
        });
        
        const handleMouseEnter = () => {
          if (bgRect) {
            bgRect.setAttribute('fill', 'black');
          }
          textElements.forEach((el) => {
            el.setAttribute('fill', 'white');
            el.setAttribute('fill-opacity', '1');
          });
        };
        
        const handleMouseLeave = () => {
          if (bgRect) {
            bgRect.setAttribute('fill', 'transparent');
          }
          textElements.forEach((el) => {
            const originalFill = originalFills.get(el);
            const originalOpacity = originalOpacities.get(el);
            if (originalFill !== null && originalFill !== undefined) {
              el.setAttribute('fill', originalFill);
            } else {
              el.removeAttribute('fill');
            }
            if (originalOpacity !== null && originalOpacity !== undefined) {
              el.setAttribute('fill-opacity', originalOpacity);
            } else {
              el.removeAttribute('fill-opacity');
            }
          });
        };
        
        box.addEventListener('mouseenter', handleMouseEnter);
        box.addEventListener('mouseleave', handleMouseLeave);
      });
    };

    setupSvg(svgContainerRef.current, false);
    setupSvg(mobileSvgContainerRef.current, true);
  }, []);

  return (
    <>
      {/* Desktop: 5-column CSS Grid layout */}
      {/* Columns: left-margin | text-content | gap | diagram | right-margin */}
      {/* At lg+ breakpoint (1024px+), diagram floats in right column */}
      {/* Below lg, falls back to mobile/tablet fixed-bottom layout */}
      <div className="hidden lg:grid mt-20 grid-cols-[minmax(16px,1fr)_minmax(300px,680px)_minmax(8px,50px)_minmax(220px,540px)_minmax(16px,1fr)]">
        {/* Left margin - empty */}
        <div />
        
        {/* Text content column */}
        <div className="min-w-0">
          {introContent && (
            <div className="mb-8">
              {introContent}
            </div>
          )}
          {modelDescriptionGDocPortionMarkdown && (
            <div className="pt-6 pb-2 prose">
              <div className="w-full overflow-hidden">
                {modelDescriptionGDocPortionMarkdown}
              </div>
            </div>
          )}
          <div
            ref={explanationContainerRef}
            className="prose scroll-p-20 overflow-hidden"
            dangerouslySetInnerHTML={{ __html: explanationsHtml }}
          />
        </div>

        {/* Gap column - empty */}
        <div />
        
        {/* Sticky diagram column */}
        <div className="mt-10">
          <div 
            className="sticky top-[10vh] rounded-lg"
            onClick={handleContainerClick}
          >
            <div ref={svgContainerRef} className="p-3">
              <ModelDiagram
                id="model-architecture-diagram-svg"
              />
            </div>
          </div>
        </div>
        
        {/* Right margin - empty */}
        <div />
      </div>

      {/* Mobile/Tablet: Content with fixed bottom diagram */}
      <div className="lg:hidden mt-12">
        {/* Text content - padding matches header (pl-6) */}
        {/* Bottom padding uses calc to account for 20vh drawer + some buffer when visible */}
        <div 
          ref={mobileContentRef}
          className={`pl-6 pr-6 ${isMobileDiagramVisible ? 'pb-[calc(20vh+40px)]' : 'pb-10'}`}
        >
          {introContent && (
            <div className="mb-8">
              {introContent}
            </div>
          )}
          {modelDescriptionGDocPortionMarkdown && (
            <div className="pt-6 pb-2 prose">
              <div className="w-full overflow-hidden">
                {modelDescriptionGDocPortionMarkdown}
              </div>
            </div>
          )}
          <div
            className="prose scroll-p-20 overflow-hidden"
            dangerouslySetInnerHTML={{ __html: explanationsHtml }}
          />
        </div>
        
        {/* Fixed bottom diagram - clamped to 20% of screen height, fades in after scrolling */}
        <div 
          className={`fixed bottom-0 left-0 right-0 bg-[#fffff8]/80 backdrop-blur-md z-40 transition-all duration-500 ease-out ${
            isMobileDiagramVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8 pointer-events-none'
          }`}
          onClick={handleContainerClick}
        >
          <div ref={mobileSvgContainerRef} className="p-2 h-[20vh] flex items-center justify-center overflow-hidden [&>svg]:max-h-full">
            <ModelDiagram
              id="model-architecture-diagram-svg-mobile"
            />
          </div>
        </div>
      </div>
    </>
  );
}
