import { Activity, Clock3, Gauge, RotateCcw } from "lucide-react";

import type { Mode, QueryResult } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

type MetricCardsProps = {
  mode: Mode;
  result: QueryResult | null;
  isLoading: boolean;
};

function formatConfidence(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  return `${Math.round(value * 100)}%`;
}

function metricValues(mode: Mode, result: QueryResult | null) {
  if (!result) {
    return {
      loop: mode === "advanced" || mode === "compare" ? "0" : "-",
      latency: "n/a",
      confidence: "n/a",
      status: "idle",
    };
  }

  if (result.mode === "compare") {
    const loop = result.advanced.loopCount ?? 0;
    const totalLatency =
      result.standard.latencyMs !== null && result.advanced.latencyMs !== null
        ? result.standard.latencyMs + result.advanced.latencyMs
        : null;
    const confidence = result.advanced.confidence ?? result.standard.confidence;
    const status =
      result.standard.status === result.advanced.status
        ? result.standard.status
        : `${result.standard.status} / ${result.advanced.status}`;

    return {
      loop: String(loop),
      latency: totalLatency === null ? "n/a" : `${totalLatency}ms`,
      confidence: formatConfidence(confidence),
      status,
    };
  }

  return {
    loop: result.loopCount === null ? "0" : String(result.loopCount),
    latency: result.latencyMs === null ? "n/a" : `${result.latencyMs}ms`,
    confidence: formatConfidence(result.confidence),
    status: result.status,
  };
}

export function MetricCards({ mode, result, isLoading }: MetricCardsProps) {
  const metrics = metricValues(mode, result);
  const loadingValue = isLoading ? "..." : undefined;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Card className="border-slate-200 shadow-sm">
        <CardContent className="flex items-start justify-between p-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Loop Count</p>
            <p className="mt-1 text-2xl font-semibold text-slate-900">{loadingValue || metrics.loop}</p>
          </div>
          <RotateCcw className="h-5 w-5 text-blue-600" />
        </CardContent>
      </Card>

      <Card className="border-slate-200 shadow-sm">
        <CardContent className="flex items-start justify-between p-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Total Latency</p>
            <p className="mt-1 text-2xl font-semibold text-slate-900">{loadingValue || metrics.latency}</p>
          </div>
          <Clock3 className="h-5 w-5 text-violet-600" />
        </CardContent>
      </Card>

      <Card className="border-slate-200 shadow-sm">
        <CardContent className="flex items-start justify-between p-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Confidence</p>
            <p className="mt-1 text-2xl font-semibold text-slate-900">{loadingValue || metrics.confidence}</p>
          </div>
          <Gauge className="h-5 w-5 text-cyan-600" />
        </CardContent>
      </Card>

      <Card className="border-slate-200 shadow-sm">
        <CardContent className="flex items-start justify-between p-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Status</p>
            <div className="mt-1">
              <Badge variant="outline" className="border-slate-300 bg-slate-100 text-slate-700">
                {loadingValue || metrics.status}
              </Badge>
            </div>
          </div>
          <Activity className="h-5 w-5 text-emerald-600" />
        </CardContent>
      </Card>
    </div>
  );
}
