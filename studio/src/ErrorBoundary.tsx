import { Component, type ErrorInfo, type ReactNode } from 'react';

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
  info: ErrorInfo | null;
};

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.setState({ info });
    console.error('[Haiku Studio renderer]', error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div style={{ minHeight: '100vh', background: '#09090b', color: '#e4e4e7', fontFamily: 'Segoe UI, Arial, sans-serif', padding: 40 }}>
        <div style={{ maxWidth: 920, margin: '8vh auto', background: '#111114', border: '1px solid #27272a', borderRadius: 18, padding: 32 }}>
          <h1 style={{ margin: '0 0 12px', fontSize: 24 }}>Haiku Studio renderer crashed</h1>
          <p style={{ color: '#a1a1aa', lineHeight: 1.55 }}>
            The desktop shell and backend started, but the React UI threw an error while rendering.
            Check <code>studio\\logs\\renderer.log</code> for the full console output.
          </p>
          <pre style={{ whiteSpace: 'pre-wrap', background: '#18181b', border: '1px solid #27272a', borderRadius: 10, padding: 14, overflowX: 'auto' }}>
            {this.state.error.stack || this.state.error.message}
            {this.state.info?.componentStack || ''}
          </pre>
        </div>
      </div>
    );
  }
}
