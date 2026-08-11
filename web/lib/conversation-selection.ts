export class ConversationSelection {
  private activeId: string | null = null;

  select(id: string | null): void {
    this.activeId = id;
  }

  isCurrent(id: string): boolean {
    return this.activeId === id;
  }
}
