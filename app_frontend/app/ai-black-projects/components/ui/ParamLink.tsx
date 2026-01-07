'use client';

/**
 * Scrolls to a parameter input in the sidebar and highlights it.
 * Expands any collapsed parent <details> elements.
 * Focuses the value display span to trigger edit mode.
 */
export function scrollToParameter(paramId: string) {
  const input = document.getElementById(paramId);
  if (input) {
    // First, expand any collapsed parent <details> elements
    let parent = input.parentElement;
    while (parent) {
      if (parent.tagName === 'DETAILS' && !(parent as HTMLDetailsElement).open) {
        (parent as HTMLDetailsElement).open = true;
      }
      parent = parent.parentElement;
    }
    // Small delay to allow DOM to update after expanding
    setTimeout(() => {
      input.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Find the value display span and focus it to trigger edit mode
      const valueSpan = document.querySelector(`[data-param-value="${paramId}"]`) as HTMLElement;
      if (valueSpan) {
        valueSpan.focus();
      } else {
        input.focus();
      }
    }, 50);
  }
}

interface ParamLinkProps {
  paramId?: string;
  children: React.ReactNode;
}

/**
 * Clickable link that scrolls to and highlights a parameter input in the sidebar.
 * Uses the .param-link CSS class for styling.
 */
export function ParamLink({ paramId, children }: ParamLinkProps) {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (paramId) {
      scrollToParameter(paramId);
    }
  };

  return (
    <span
      className="param-link"
      onClick={handleClick}
      style={{
        cursor: paramId ? 'pointer' : 'default',
        textDecoration: paramId ? 'underline' : 'none',
      }}
    >
      {children}
    </span>
  );
}

export default ParamLink;
