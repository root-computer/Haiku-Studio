import React from 'react';

export const HuggingFaceIcon = ({ className, ...props }: React.SVGProps<SVGSVGElement>) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    {...props}
  >
    {/* Face Outline */}
    <circle cx="12" cy="12" r="9" />
    {/* Eyes */}
    <path d="M9 11c.5 0 1-.5 1-1" />
    <path d="M15 11c-.5 0-1-.5-1-1" />
    {/* Smile */}
    <path d="M8 15c1 1.5 3 2 4 2s3-.5 4-2" />
    {/* Hugging Hands */}
    <path d="M4.5 14c-1 0-1.5 1-1.5 2s1 1.5 2 1.5h1" />
    <path d="M19.5 14c1 0 1.5 1 1.5 2s-1 1.5-2 1.5h-1" />
  </svg>
);
