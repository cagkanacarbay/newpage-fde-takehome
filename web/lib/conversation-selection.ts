type ConversationLoad = {
  id: string | null;
  generation: number;
};

export class ConversationSelection {
  private activeId: string | null = null;
  private generation = 0;

  select(id: string | null): ConversationLoad {
    this.activeId = id;
    this.generation += 1;
    return { id, generation: this.generation };
  }

  isCurrent(load: ConversationLoad): boolean {
    return this.activeId === load.id && this.generation === load.generation;
  }
}

export class ConversationListRefresh {
  private generation = 0;

  start(): number {
    this.generation += 1;
    return this.generation;
  }

  isCurrent(generation: number): boolean {
    return this.generation === generation;
  }
}
