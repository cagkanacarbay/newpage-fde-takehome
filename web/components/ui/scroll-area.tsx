import * as React from "react";

import { cn } from "@/lib/utils";

const ScrollArea = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div">
>(({ className, ...props }, ref) => (
  <div
    className={cn("overflow-y-auto overscroll-contain", className)}
    ref={ref}
    {...props}
  />
));
ScrollArea.displayName = "ScrollArea";

export { ScrollArea };
