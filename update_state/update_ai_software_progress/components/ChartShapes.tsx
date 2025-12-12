import { ShapeProps } from '@/app/types';

export const createDiamondShape = (fill: string, stroke: string = '#ffffff') => {
  const DiamondShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <polygon
        points={`${cx},${cy - 6} ${cx + 6},${cy} ${cx},${cy + 6} ${cx - 6},${cy}`}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  DiamondShape.displayName = 'DiamondShape';
  return DiamondShape;
};

export const createSquareShape = (fill: string, stroke: string = '#ffffff') => {
  const SquareShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <rect
        x={cx - 5}
        y={cy - 5}
        width={10}
        height={10}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  SquareShape.displayName = 'SquareShape';
  return SquareShape;
};

export const createTriangleShape = (fill: string, stroke: string = '#ffffff') => {
  const TriangleShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <polygon
        points={`${cx},${cy - 6} ${cx + 6},${cy + 6} ${cx - 6},${cy + 6}`}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  TriangleShape.displayName = 'TriangleShape';
  return TriangleShape;
};

export const createStarShape = (fill: string, stroke: string = '#ffffff') => {
  const StarShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    const r1 = 6, r2 = 3;
    const points = [];
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? r1 : r2;
      const angle = (i * Math.PI) / 5 - Math.PI / 2;
      points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return (
      <polygon
        points={points.join(' ')}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  StarShape.displayName = 'StarShape';
  return StarShape;
};

export const createCircleShape = (fill: string, stroke: string = '#ffffff') => {
  const CircleShape = (props: ShapeProps) => {
    const { cx, cy, onMouseEnter, onMouseLeave, style } = props;
    return (
      <circle
        cx={cx}
        cy={cy}
        r={5}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        style={onMouseEnter ? { cursor: 'pointer', ...style } : style}
      />
    );
  };
  CircleShape.displayName = 'CircleShape';
  return CircleShape;
};