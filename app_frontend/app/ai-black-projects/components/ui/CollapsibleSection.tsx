'use client';

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
}: CollapsibleSectionProps) {
  return (
    <details
      className="mb-3 pt-1 rounded-none !m-0 !px-0 border-t border-b-0 border-l-0 border-r-0 border-gray-300"
      open={defaultOpen}
    >
      <summary className="text-[11px] font-bold cursor-pointer py-0.5">{title}</summary>
      <div className="mt-2">{children}</div>
    </details>
  );
}
