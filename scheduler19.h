#ifndef LSG_SCHEDULER
#define LSG_SCHEDULER

struct Scheduler {
	Scheduler();
	~Scheduler();

	void init(unsigned ntasks);
	void initData(unsigned data = 0);
	void initOrder();
	void update(unsigned task, unsigned newdata);
	unsigned next();
	unsigned prev();
	void letsOrder();	// using current data.
	void swapOrder(unsigned one, unsigned two);
	void swapEnabled(unsigned one, unsigned two);
	void print();
	void reorderOnUpdate();
	void dontReorderOnUpdate();
	void reorderEnabled();

	void initEnabled();
	void disableAll();
	void enableAll();
	void assignEnabledAll(bool state);
	void disable(unsigned task);
	void enable(unsigned task);
	void assignEnabled(unsigned task, bool state);
	bool isEnabled(unsigned task);
	bool isDisabled(unsigned task);
	unsigned findEnabled();
	unsigned nextEnabled();
	void printEnabled();

	unsigned *data, *order;
	unsigned ntasks;
	unsigned currenttask, previousupdated;
	bool reorderonupdate;

	bool *enabled; 
	unsigned *enabledinuse;
	unsigned nenabled;
};
Scheduler::Scheduler() {
	ntasks = 0;
	data = order = NULL;
	enabled = NULL;
	enabledinuse = NULL;
	currenttask = 0;
	previousupdated = 0;
	reorderonupdate = false;
	nenabled = 0;
}
Scheduler::~Scheduler() {
	free(data);
	free(order);
	free(enabled);
	free(enabledinuse);
}
void Scheduler::init(unsigned ntasks) {
	data = (unsigned *)malloc(ntasks * sizeof(unsigned));
	order = (unsigned *)malloc(ntasks * sizeof(unsigned));
	enabled = (bool *)malloc(ntasks * sizeof(bool));
	enabledinuse = (unsigned *)malloc(ntasks * sizeof(unsigned));
	this->ntasks = ntasks;
	initData(1);	// 1 to ensure progress.
	initOrder();
	initEnabled();
	letsOrder();
}
void Scheduler::letsOrder() {
	for (unsigned ii = 0; ii < ntasks - 1; ++ii) {
		for (unsigned jj = ii + 1; jj < ntasks; ++jj) {
			if (data[order[ii]] < data[order[jj]]) {
				swapOrder(ii, jj);
			}
		}
	}
	currenttask = 0;
}
void Scheduler::initEnabled() {
	disableAll();
}
void Scheduler::disableAll() {
	assignEnabledAll(false);
	nenabled = 0;
}
void Scheduler::enableAll() {
	assignEnabledAll(true);
	nenabled = ntasks;
}
void Scheduler::assignEnabledAll(bool state) {
	for (unsigned ii = 0; ii < ntasks; ++ii) {
		assignEnabled(ii, state);
	}
}
void Scheduler::enable(unsigned task) {
	if (isDisabled(task)) {
		++nenabled;
		assignEnabled(task, true);
	}
}
void Scheduler::disable(unsigned task) {
	if (isEnabled(task)) {
		--nenabled;
		assignEnabled(task, false);
	}
}
void Scheduler::assignEnabled(unsigned task, bool state) {
	enabled[task] = state;
}
bool Scheduler::isEnabled(unsigned task) {
	return enabled[task];
}
bool Scheduler::isDisabled(unsigned task) {
	return !isEnabled(task);
}
void Scheduler::swapOrder(unsigned one, unsigned two) {
	unsigned tmp = order[one];
	order[one] = order[two];
	order[two] = tmp;
}
void Scheduler::swapEnabled(unsigned one, unsigned two) {
	unsigned tmp = enabledinuse[one];
	enabledinuse[one] = enabledinuse[two];
	enabledinuse[two] = tmp;
}
void Scheduler::initData(unsigned newdata/* = 0*/) {
	for (unsigned ii = 0; ii < ntasks; ++ii) {
		data[ii] = newdata;
	}
}
void Scheduler::initOrder() {
	for (unsigned ii = 0; ii < ntasks; ++ii) {
		order[ii] = ii;
	}
}
void Scheduler::update(unsigned task, unsigned newdata) {
	if (task < ntasks) {
		data[task] = newdata;
		if (reorderonupdate) {
			letsOrder();
		}
		if (newdata) {
			previousupdated = task;
		}
		return;
	}
	printf("Error: %s(%d): Task %d out of bound %d.\n", __FILE__, __LINE__, task, ntasks);
}
unsigned Scheduler::next() {
	unsigned retval = order[currenttask++];
	if (currenttask == ntasks) {
		letsOrder();
	}
	return retval;
}
unsigned Scheduler::prev() {
	/*if (currenttask == 0 || currenttask == 1) {
		return 0;	// don't know.
	}
	return order[currenttask - 2];*/
	return previousupdated;
}
void Scheduler::print() {
	for (unsigned ii = 0; ii < ntasks; ++ii) {
		printf("%3d ", order[ii]);
	}
	printf("\n");
	for (unsigned ii = 0; ii < ntasks; ++ii) {
		printf("%3d ", data[order[ii]]);
	}
	printf("\n");
}
void Scheduler::printEnabled() {
	for (unsigned ii = 0; ii < nenabled; ++ii) {
		printf("%3d ", enabledinuse[ii]);
	}
	printf("\n");
}
void Scheduler::reorderOnUpdate() {
	reorderonupdate = true;
}
void Scheduler::dontReorderOnUpdate() {
	reorderonupdate = false;
}
unsigned Scheduler::findEnabled() {
	unsigned len = 0;
	for (unsigned ii = 0; ii < ntasks; ++ii) {
		if (isEnabled(ii)) {
			enabledinuse[len++] = ii;
		}
	}
	currenttask = 0;
	if (len != nenabled) {
		printf("Error: %s(%d): len(%d) and nenabled(%d) don't match.\n", __FILE__, __LINE__, len, nenabled);
	}
	return nenabled;
}
unsigned Scheduler::nextEnabled() {
	return enabledinuse[currenttask++];
}
void Scheduler::reorderEnabled() {
	for (unsigned ii = 0; ii < nenabled - 1; ++ii) {
		for (unsigned jj = ii + 1; jj < nenabled; ++jj) {
			if (data[enabledinuse[ii]] < data[enabledinuse[jj]]) {
				swapEnabled(ii, jj);
			}
		}
	}

}
#endif
