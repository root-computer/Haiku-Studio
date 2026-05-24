import React from 'react';

type MotionOnlyProps = {
  initial?: unknown;
  animate?: unknown;
  exit?: unknown;
  transition?: unknown;
  variants?: unknown;
  layout?: unknown;
  whileHover?: unknown;
  whileTap?: unknown;
  drag?: unknown;
  dragConstraints?: unknown;
};

function stripMotionProps<T extends object>(props: T & MotionOnlyProps) {
  const {
    initial,
    animate,
    exit,
    transition,
    variants,
    layout,
    whileHover,
    whileTap,
    drag,
    dragConstraints,
    ...rest
  } = props;
  return rest;
}

type DivProps = React.HTMLAttributes<HTMLDivElement> & MotionOnlyProps;

const MotionDiv = React.forwardRef<HTMLDivElement, DivProps>((props, ref) => {
  return <div ref={ref} {...stripMotionProps(props)} />;
});
MotionDiv.displayName = 'MotionDiv';

export const motion = {
  div: MotionDiv,
};

export function AnimatePresence({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
