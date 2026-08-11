"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/utils";

const Sheet = Dialog.Root;
const SheetTrigger = Dialog.Trigger;
const SheetClose = Dialog.Close;

type SheetContentProps = React.ComponentPropsWithoutRef<typeof Dialog.Content> & {
  side?: "left" | "full";
  hideClose?: boolean;
};

/**
 * shadcn-style sheet on the NewPage system: no border, no shadow, depth from
 * fill + big radius.
 */
const SheetContent = React.forwardRef<
  React.ComponentRef<typeof Dialog.Content>,
  SheetContentProps
>(({ className, children, side = "left", hideClose = false, ...props }, ref) => (
  <Dialog.Portal>
    <Dialog.Overlay className="fixed inset-0 z-40 bg-ink/30" />
    <Dialog.Content
      ref={ref}
      className={cn(
        "fixed z-50 flex flex-col bg-surface focus:outline-none",
        side === "left" && "inset-y-2 left-2 w-[300px] rounded-r-3xl",
        side === "full" && "inset-0",
        className,
      )}
      {...props}
    >
      {children}
      {!hideClose ? (
        <Dialog.Close className="absolute right-4 top-4 rounded-full p-2 text-faint transition-colors hover:bg-surface-2 hover:text-body focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal">
          <X className="size-5" />
          <span className="sr-only">Close</span>
        </Dialog.Close>
      ) : null}
    </Dialog.Content>
  </Dialog.Portal>
));
SheetContent.displayName = "SheetContent";

export { Sheet, SheetClose, SheetContent, SheetTrigger };
