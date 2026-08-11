/**
 * Unsent composer drafts, kept per conversation id in memory. Drafts are
 * deliberately not persisted: a refresh loses them.
 */
export class DraftStore {
  private readonly drafts = new Map<string, string>();

  get(conversationKey: string): string {
    return this.drafts.get(conversationKey) ?? "";
  }

  set(conversationKey: string, text: string): void {
    if (text === "") {
      this.drafts.delete(conversationKey);
      return;
    }
    this.drafts.set(conversationKey, text);
  }
}
