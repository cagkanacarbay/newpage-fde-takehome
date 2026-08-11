import * as React from "react";

import { cn } from "@/lib/utils";

const Input = React.forwardRef<
  HTMLInputElement,
  React.ComponentProps<"input">
>(({ className, type, ...props }, ref) => (
  <input
    type={type}
    className={cn(
      "flex h-11 w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-base text-zinc-950 outline-none placeholder:text-zinc-500 focus-visible:ring-2 focus-visible:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
      className,
    )}
    ref={ref}
    {...props}
  />
));
Input.displayName = "Input";

export { Input };
